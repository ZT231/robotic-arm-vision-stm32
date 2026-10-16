#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5 视觉引导抓取主程序：识别 -> 像素 -> 毫米 -> IK -> 串口 -> 机械臂抓取。

完整链路：
  摄像头/图片
    -> Detector.detect()            识别目标，得像素中心 (px,py)
    -> pixel_to_world(H,px,py)      像素 -> 工作台毫米 (wx,wy)
    -> 减底座偏移 base_x/base_y     毫米 -> 机械臂坐标 (x,y) cm
    -> ArmIK.solve(x,y)             机械臂坐标 -> 6 关节角度
    -> ArmIK.to_pulses(angles)      角度 -> 舵机脉宽
    -> ArmHost.joint(*pulses)       串口下发，机械臂执行

用法（在 vision 目录下）：
  python grasp.py --source 0 --homography homography.json --sim
  python grasp.py --source 0 --homography homography.json --port COM3 --base-x 10 --base-y 8
  python grasp.py --source 0 --homography homography.json --port COM3 --mode 2
  python grasp.py --source 0 --homography homography_hand.json --port COM3 --scan   # 眼在手：检测前先到扫描姿态

运行模式（启动时选择，或 --mode 指定）：
  1 = 逐步抓取（每步按空格）
  2 = 自动连续抓取（对画面目标做快照，依次自动抓取+放物）
  3 = 只抓指定颜色（输入颜色名）
  4 = 鼠标点击模式（左键点目标自动抓取；抓完左键点工作区选投放点放物）

鼠标（模式 4）：
  左键  空闲时点击画面中的目标 -> 自动抓取（红圈标记所选目标）；
        抓取完成后点击工作区任意点 -> 放到该点（蓝圈标记投放点）

按键：
  空格  模式1（逐步）：抓取 4 步（准备 -> 下降 -> 闭合 -> 抬起）每步一按；
        抓完后按一次空格 = 放物一步（保持闭合降到位 -> 原地张开 -> 空爪抬起，
        内部自动等待舵机到位和物体落稳）；
        再按一次空格选择下一个目标并开始下一轮抓取
        模式2（自动）：自动抓取结束后按空格重新开始自动抓取
  h     回 HOME（自动模式中：先放物再回位并停止自动）
  c     切换颜色过滤：指定过颜色（--color/模式3）后，模式1/2 中按 c 取消/恢复过滤
  r     返回选择菜单（重新选择运行模式；若有物体在爪里会先安全放物）
  q     退出

远目标自动前倾：目标超出垂直抓取范围时，自动逐步前倾夹爪
（每次 5°，最大 --tilt-max 度）扩大臂展，四步共用同一前倾角。

按形状沿边抓取：方块/长方形/多边形检测旋转外接矩形的长边方向，J5 自动旋转
夹爪使其两指垂直于长边（贴住两条长边、跨过宽度），避免抓到对角被弹飞；
圆形保持默认方向。J5 基准与视觉角度的对应关系用 --wrist-offset 校准。
"""

import argparse
import json
import math
import os
import sys
import time

import cv2

try:
    import serial
except ImportError:
    print("缺少依赖 pyserial，请先执行： pip install pyserial")
    sys.exit(1)

from detector import Detector, draw_targets
from homography import load_homography, pixel_to_world, world_to_pixel
from ik import ArmIK, JOINTS

PREP_Z = 5.0    # 准备/抬起高度（夹爪离台面 cm）。目标较远时过高会超出臂展不可达；物体较高时可调大
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def letterbox(img, w, h):
    """把 img 等比例缩放到 w×h 内（长宽同比例），黑边填充，窗口任意尺寸画面不变形。"""
    fh, fw = img.shape[:2]
    if fw <= 0 or fh <= 0 or w <= 0 or h <= 0:
        return img
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def draw_dashed_rect(img, x, y, w, h, color, dash=10, gap=6, thickness=2):
    """在 img 上画虚线矩形（ROI 框），坐标为原始画面像素。"""
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        L = max(1, abs(x1 - x0) + abs(y1 - y0))
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        pos = 0.0
        while pos < L:
            ax, ay = x0 + ux * pos, y0 + uy * pos
            bx, by = x0 + ux * min(pos + dash, L), y0 + uy * min(pos + dash, L)
            cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)
            pos += dash + gap


def open_camera(source):
    if os.path.splitext(str(source))[1].lower() in IMG_EXTS:
        img = cv2.imread(source)
        if img is None:
            raise ValueError("无法读取图片 %s" % source)
        return None, img

    idx = int(source) if str(source).isdigit() else source
    if isinstance(idx, int):
        # Windows 用 DSHOW 后端：MSMF 在该机型上会导致 cap.read() 永久卡死
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        except Exception:
            cap = None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(idx)  # 回退默认后端
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise ValueError("无法打开摄像头 %s" % source)

    # 预热：确认能真正读到画面，防止打开成功但 read() 无响应
    for _ in range(8):
        ok, frame = cap.read()
        if ok:
            return cap, None
        time.sleep(0.2)
    cap.release()
    raise ValueError("摄像头 %s 无法读取画面（请检查是否被其他程序占用）" % source)


STEP_NAMES = ["准备", "下降", "闭合", "抬起"]
DROP_NAMES = ["降到位", "放开", "抬起"]
TILT_STEP = 5.0   # 自动前倾的步进(度)
VALID_COLORS = ("red", "green", "blue", "yellow", "pink")


class _EmaSmoother:
    """帧间 EMA 平滑：抑制检测框中心/方向角的像素级抖动（显示与抓取都用平滑值）。

    匹配规则：同颜色+同形状、与上帧最近的目标（距离 < match_px）才做指数平滑；
    目标快速移动或新出现时直接用新值，避免滞后拖尾。
    """

    def __init__(self, alpha=0.4, match_px=60.0):
        self.a = alpha
        self.match_px2 = match_px * match_px
        self.prev = []

    def __call__(self, targets):
        if not targets:
            self.prev = []
            return targets
        out = []
        for t in targets:
            best, best_d = None, 1e18
            for p in self.prev:
                if p.color != t.color or p.shape != t.shape:
                    continue
                d = (p.cx - t.cx) ** 2 + (p.cy - t.cy) ** 2
                if d < best_d:
                    best, best_d = p, d
            if best is not None and best_d < self.match_px2:
                a = self.a
                t.cx = a * t.cx + (1 - a) * best.cx
                t.cy = a * t.cy + (1 - a) * best.cy
                # w/h 平滑：正方形 w≈h 时临界抖动会让长边方向每帧跳 90°，平滑后判断稳定
                t.w = a * t.w + (1 - a) * best.w
                t.h = a * t.h + (1 - a) * best.h
                # minAreaRect 角度以 180° 为周期，取环形最短差插值，避免跨越跳变
                diff = (t.angle - best.angle) % 180.0
                if diff > 90.0:
                    diff -= 180.0
                t.angle = best.angle + a * diff
                # 轮廓顶点平滑（框贴合轮廓绘制）：顶点数相同且按起点对齐后距离小才平滑，
                # 否则直接用新帧顶点（分类跳变/轮廓变形时避免拖尾）
                if (t.vertices and best.vertices
                        and len(t.vertices) == len(best.vertices) >= 3):
                    nv = len(t.vertices)
                    cur, old = t.vertices, best.vertices
                    best_off, best_sum = 0, 1e18
                    for off in range(nv):
                        s = sum((cur[(i + off) % nv][0] - old[i][0]) ** 2
                                + (cur[(i + off) % nv][1] - old[i][1]) ** 2
                                for i in range(nv))
                        if s < best_sum:
                            best_sum, best_off = s, off
                    if best_sum < self.match_px2 * nv:
                        aligned = [cur[(i + best_off) % nv] for i in range(nv)]
                        t.vertices = [
                            (a * aligned[i][0] + (1 - a) * old[i][0],
                             a * aligned[i][1] + (1 - a) * old[i][1])
                            for i in range(nv)
                        ]
            out.append(t)
        self.prev = out
        return targets


def choose_color(prompt="颜色 (red/green/blue/yellow/pink): "):
    """交互输入颜色名并校验合法性；非法输入时提示并重新询问。"""
    while True:
        try:
            c = (input(prompt).strip() or "red").lower()
        except EOFError:
            c = "red"
        if c in VALID_COLORS:
            return c
        print("无效颜色 '%s'，可选: %s" % (c, ", ".join(VALID_COLORS)))


def _check_no_clamp(ik, solves, pulses, tilt):
    """校验脉宽反解角度与 IK 输出一致；不一致说明关节角度越界被夹紧，末端到不了位。

    to_pulses 对越界的 J3/J4 角度会静默夹紧到标定表端点，此时机械臂实际到不了
    目标点。这里反解回来对比，发现夹紧就把该 tilt 判为不可用。"""
    for ang, p in zip(solves, pulses):
        servo = ik.to_angles(p)
        expect = [ang[0], ang[1], ang[2] - ang[1], -180.0 - ang[2] + tilt, ang[4], ang[5]]
        for name, got, exp in zip(JOINTS, servo, expect):
            # 容差 5°：脉宽取整仅引入 ~0.06° 误差；夹紧会偏差几十度，足够区分
            if abs(got - exp) > 5.0:
                raise ValueError("%s 角度被限位夹紧(%.1f!=%.1f)，前倾 %.0f° 不可用"
                                 % (name, got, exp, tilt))


def _axis_th1(ik, x, y, wrist):
    """机械臂实际 J1 方位角(物理度)：与 ik.solve 同约定做腕部偏心修正。

    solve 内偏心补偿后 J1 基于 J5 轴点而非目标点；若 calc_wrist 直接用目标点
    方位，与机械臂实际 J1 会有位置相关偏差(偏心1cm、r=18cm 时约3°)，导致沿边
    方向在不同位置偏不同角度。偏心方向 ψ 依赖 wrist，用目标方位近似后迭代一次
    即可收敛（偏心<2cm）。"""
    th1 = math.degrees(math.atan2(y, x)) - 90.0
    if ik.wrist_ecc > 1e-6:
        psi = math.radians(math.degrees(math.atan2(y, x)) + wrist + ik.ecc_dir)
        ec = ik.wrist_ecc   # tilt 未知：倾斜抓取时偏心水平分量按 cos(tilt) 衰减，误差更小
        xa, ya = x - ec * math.cos(psi), y - ec * math.sin(psi)
        th1 = math.degrees(math.atan2(ya, xa)) - 90.0
    return (th1 + 180.0) % 360.0 - 180.0


def calc_wrist(ik, H, t, x, y, offset=0.0, grip_conv=0.0):
    """按目标物形状计算 J5 夹爪张开方向角(度)，沿边抓取避免对角弹飞。

    正方形沿任一边、长方形垂直长边（两指贴住两条长边、跨过宽度）。
    仅 square/rectangle/diamond 计算方向（沿边抓取有意义）；
    圆形/三角形/五边形/多边形等保持默认方向（垂直长边抓三角形会落在顶点上，易滑动）。
    链路：
      长边方向(像素, OpenCV angle 为 w 边方向) -> 用 H 变换长边端点得世界角 a_world
      -> J5=0 时张开方向沿臂前向(th1+90°) -> wrist = a_world - th1 + 校准（两指即垂直长边）。
    注：J1 方位角 th1 只由 (x,y) 决定（与 z/tilt/wrist 无关），直接用与 ik.solve
    相同的公式计算，避免依赖 solve 成功。倾斜抓取(tilt>0)时方向精度略降，可接受。"""
    if t.shape not in ("square", "rectangle", "diamond"):
        return 0.0
    a = math.radians(t.angle)
    if t.w >= t.h:
        long_dir = (math.cos(a), math.sin(a))
        half = t.w / 2.0
    else:
        long_dir = (math.cos(a + math.pi / 2.0), math.sin(a + math.pi / 2.0))
        half = t.h / 2.0
    w1 = pixel_to_world(H, t.cx, t.cy)
    w2 = pixel_to_world(H, t.cx + half * long_dir[0], t.cy + half * long_dir[1])
    a_world = math.degrees(math.atan2(w2[1] - w1[1], w2[0] - w1[0]))
    # J5=0 时两指连线相对臂前向偏 grip_conv（结构约定角，真机实测标定：
    # 默认 0=J5=0 张开沿臂前向；若真机压短边（差 90°），试 grip_conv=±90）。
    # 张开方向(世界) = 臂前向(th1+90°) + grip_conv + wrist。
    # 要两指垂直长边（贴住两条长边）-> 张开方向 = a_world + 90°
    # -> wrist = a_world + 90° - (th1+90°) - grip_conv + offset = a_world - th1 - grip_conv + offset。
    # th1 用轴点方位（与 ik.solve 的偏心补偿一致）：先以目标方位得 wrist 初值，再迭代一次。
    th1 = math.degrees(math.atan2(y, x)) - 90.0
    wrist = a_world - th1 - grip_conv + offset
    wrist = a_world - _axis_th1(ik, x, y, wrist) - grip_conv + offset
    j5 = ik.joints["J5"]
    # 注意 J5 是反向舵机（实测 map 降序：deg_min > deg_max），范围必须归一化
    lo, hi = float(j5["deg_min"]), float(j5["deg_max"])
    lo, hi = min(lo, hi), max(lo, hi)
    # 张开方向是“直线”语义，wrist ±180° 完全等价（两指压的还是同一对边）。
    # 但偏心模型下 ψ = 轴方位 + wrist + ecc_dir，wrist 翻转 180° 会使偏心方向
    # 反向、轴点/J1 微变，张开方向会偏 (e/r) 级角度。故等价角不按数值最近选，
    # 而按「实际张开方向(用 _axis_th1 模拟 solve) 最接近期望」选。
    def _line_diff(a, b):
        d = abs((a - b) % 180.0)
        return min(d, 180.0 - d)
    cands = [c for c in (wrist, wrist + 180.0, wrist - 180.0) if lo <= c <= hi]
    if cands:
        expect_open = a_world + 90.0 + offset
        return min(cands, key=lambda c: _line_diff(expect_open, _axis_th1(ik, x, y, c) + 90.0 + grip_conv + c))
    return max(lo, min(hi, wrist))


def grab_pulses(ik, args, x, y, wrist=0.0):
    """计算四段抓取动作的脉宽序列；不可达返回 (None, 原因, None)。

    目标较远时先尝试垂直抓取(tilt=0)，不可达则逐步前倾夹爪扩大臂展，
    直至可达或达到 --tilt-max 上限。四段共用同一 tilt 和 wrist，保证轨迹一致。
    z 取 grab_height/--prep-z 叠加 --off-z 高度偏移。
    返回 (pulses, err, tilt)，err 为 None 表示成功。"""
    z_grab = ik.grab_h + args.off_z
    z_prep = args.prep_z + args.off_z
    tilt = 0.0
    while tilt <= args.tilt_max + 1e-9:
        try:
            solves = [
                ik.solve(x, y, z=z_prep, claw=args.grip_open, tilt=tilt, wrist=wrist),   # 准备
                ik.solve(x, y, z=z_grab, claw=args.grip_open, tilt=tilt, wrist=wrist),   # 下降
                ik.solve(x, y, z=z_grab, claw=args.grip_close, tilt=tilt, wrist=wrist),  # 闭合
                ik.solve(x, y, z=z_prep, claw=args.grip_close, tilt=tilt, wrist=wrist),  # 抬起
            ]
            pulses = [ik.to_pulses(a) for a in solves]
            _check_no_clamp(ik, solves, pulses, tilt)
            return pulses, None, tilt
        except ValueError as e:
            msg = str(e)
            # 过近/重合：前倾只会让手腕更近，直接放弃；J5/J6 越界：与前倾无关，直接放弃
            if "过近" in msg or "重合" in msg or "J5" in msg or "J6" in msg:
                return None, msg, None
            tilt += TILT_STEP
    return None, "不可达: 前倾到 %.0f° 仍超出可达范围" % args.tilt_max, None


def do_step(host, pulses, name):
    if host is None:
        print("  [sim] %s %s" % (name, pulses))
    else:
        r = host.joint(*pulses)
        print("  [%s] %s -> %r" % (name, pulses, r))


def wait_auto(host, seconds):
    """自动模式步骤间等待：真机等 seconds 让舵机到位；sim 只等 0.2s 便于快速观察。"""
    time.sleep(min(seconds, 0.2) if host is None else seconds)


def release_pulses(host, ik, args, hold):
    """放物三步（一步完成）：保持闭合降到抓取高度 -> 原地张开 -> 空爪抬起。
    物体全程在爪里直到贴着台面才松手，避免下降途中掉落或移动甩出。"""
    grip_close = ik.to_pulses([0, 0, 0, 0, 0, args.grip_close])[5]
    j6_open = ik.to_pulses([0, 0, 0, 0, 0, args.grip_open])[5]
    do_step(host, hold[1][:5] + [grip_close], "降到位")
    wait_auto(host, args.drop_wait)          # 等机械臂降到抓取高度
    do_step(host, hold[1], "放开")            # 原地张开：物体贴着台面松手
    wait_auto(host, args.release_wait)       # 等物体落稳
    do_step(host, hold[3][:5] + [j6_open], "抬起")   # 空爪抬起


def drop_pulses(ik, args, x, y, wrist):
    """解算"放到指定点"的三段脉宽：降到位(闭合) -> 原地张开 -> 抬起(张开)。
    与 grab_pulses 一样，目标较远时逐步前倾扩大臂展；返回 (seq, err, tilt)。"""
    z_grab = ik.grab_h + args.off_z
    z_prep = args.prep_z + args.off_z
    tilt = 0.0
    while tilt <= args.tilt_max + 1e-9:
        try:
            down = ik.solve(x, y, z=z_grab, claw=args.grip_close, tilt=tilt, wrist=wrist)
            mid = list(down)
            mid[5] = args.grip_open
            up = ik.solve(x, y, z=z_prep, claw=args.grip_open, tilt=tilt, wrist=wrist)
            seq = [ik.to_pulses(down), ik.to_pulses(mid), ik.to_pulses(up)]
            _check_no_clamp(ik, [down, mid, up], seq, tilt)
            return seq, None, tilt
        except ValueError as e:
            msg = str(e)
            if "过近" in msg or "重合" in msg or "J5" in msg or "J6" in msg:
                return None, msg, None
            tilt += TILT_STEP
    return None, "投放点不可达: 前倾到 %.0f° 仍超出可达范围" % args.tilt_max, None


def choose_mode():
    """交互选择运行模式，返回 1~4；非交互环境（stdin 关闭）回退默认 1。"""
    print("\n运行模式：")
    print("  1 = 逐步抓取（每步按空格）")
    print("  2 = 自动连续抓取（依次抓完画面所有目标）")
    print("  3 = 只抓指定颜色（输入颜色名）")
    print("  4 = 鼠标点击模式（左键点目标抓取，抓完点工作区选投放点）")
    try:
        k = input("选择 (回车=1) > ").strip() or "1"
    except EOFError:
        k = "1"
    return int(k) if k.isdigit() and int(k) in (1, 2, 3, 4) else 1


def main():
    ap = argparse.ArgumentParser(description="视觉引导抓取主程序 (S5)")
    ap.add_argument("--source", default="0", help="摄像头编号或图片路径")
    ap.add_argument("--homography", default="homography.json", help="单应矩阵文件")
    ap.add_argument("--port", default=None, help="串口（如 COM3），缺省为 sim 演示模式")
    ap.add_argument("--sim", action="store_true", help="sim 演示模式（强制不连接串口）")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--base-x", type=float, default=None, help="J1 转轴在视觉毫米坐标系的 x（cm），缺省读 config.json")
    ap.add_argument("--base-y", type=float, default=None, help="J1 转轴在视觉毫米坐标系的 y（cm），缺省读 config.json")
    ap.add_argument("--color", default=None, help="只抓指定颜色: red/green/blue/yellow/pink")
    ap.add_argument("--grip-open", type=float, default=50.0, help="夹爪张开角度(度)，默认 50=张开最大(900us)")
    ap.add_argument("--grip-close", type=float, default=-25.0, help="夹爪闭合角度(度)，默认 -25=闭合约1900us(减小夹持力度)；-40=完全闭合2100us")
    ap.add_argument("--prep-z", type=float, default=PREP_Z, help="准备/抬起高度(cm)，默认 5.0")
    ap.add_argument("--tilt-max", type=float, default=45.0, help="远目标自动前倾最大角度(度)，默认 45")
    ap.add_argument("--off-x", type=float, default=-0.5, help="抓取点台面偏移 x(cm)，正=右 负=左")
    ap.add_argument("--off-y", type=float, default=1.5, help="抓取点台面偏移 y(cm)，正=前 负=后")
    ap.add_argument("--off-z", type=float, default=0.0, help="抓取点高度偏移 z(cm)，正=更高 负=更低")
    ap.add_argument("--wrist-offset", type=float, default=None,
                    help="J5 夹爪方向基准校准(度)：按形状自动旋转夹爪后仍不对齐时微调；缺省读 config.json 的 wrist_offset，否则 0")
    ap.add_argument("--grip-conv", type=float, default=90.0,
                    help="J5=0 时两指连线相对臂前向的结构偏角(度)，默认 0（张开沿臂前向）；"
                         "真机压短边(差 90°)时试 90 或 -90")
    ap.add_argument("--wrist-ecc", type=float, default=None,
                    help="夹取点相对 J5 轴横向偏心(cm)，缺省读 arm_params.json（默认 1.0）")
    ap.add_argument("--ecc-dir", type=float, default=None,
                    help="J5=0 时偏心方向相对臂前向的偏角(度)，缺省读 arm_params.json（默认 0=沿臂前向）")
    ap.add_argument("--release-wait", type=float, default=0.2,
                    help="放物落稳等待(秒)：低位张开后等物体落稳再抬起，默认 0.2")
    ap.add_argument("--drop-wait", type=float, default=0.2,
                    help="降到位等待(秒)：保持闭合降到抓取高度后、张开前的等待，默认 0.2")
    ap.add_argument("--mode", type=int, choices=[1, 2, 3, 4], default=None,
                    help="运行模式：1=逐步抓取 2=自动连续抓取 3=只抓指定颜色 4=鼠标点击模式；缺省启动时询问")
    ap.add_argument("--step-wait", type=float, default=1.0,
                    help="自动模式抓取步骤间等待(秒)，默认 1.0")
    ap.add_argument("--click-wait", type=float, default=1.0,
                    help="鼠标点击模式每步等待(秒)，默认 1.0；动作太急促就调大、太拖沓就调小")
    ap.add_argument("--scan", action="store_true",
                    help="眼在手：每次检测前先走到扫描姿态（姿态用 --scan-pulses 或 config.json 的 scan_pulses）")
    ap.add_argument("--scan-pulses", default=None,
                    help="扫描姿态 6 路脉宽,逗号分隔（如 1550,1502,1613,520,1421,900）；缺省读 config.json")
    args = ap.parse_args()

    # 运行模式：--mode 直接指定，否则启动时询问
    if args.mode is None:
        args.mode = choose_mode()
    if args.color and args.color not in VALID_COLORS:
        print("警告: 无效颜色名 '%s'，可选: %s；本次不限制颜色" % (args.color, ", ".join(VALID_COLORS)))
        args.color = None
    if args.mode == 3:
        if not args.color:
            args.color = choose_color()
        args.mode = 1   # 模式3 = 指定颜色后按逐步抓取
    print("运行模式: %s%s" % (
        "自动连续抓取" if args.mode == 2 else ("鼠标点击模式" if args.mode == 4 else "逐步抓取"),
        "（只抓 %s）" % args.color if args.color else ""))

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cfg = {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass
    if args.base_x is None:
        args.base_x = float(cfg.get("base_x", 0.0))
    if args.base_y is None:
        args.base_y = float(cfg.get("base_y", 0.0))
    if args.wrist_offset is None:
        args.wrist_offset = float(cfg.get("wrist_offset", 0.0))
    roi = cfg.get("roi")
    args.roi = [int(v) for v in roi] if isinstance(roi, list) and len(roi) == 4 else None
    print("base 偏移: x=%.1fcm y=%.1fcm（config.json，可用 --base-x/--base-y 覆盖）"
          % (args.base_x, args.base_y))
    print("wrist 偏移: %.1f°（--wrist-offset 覆盖，真机上 [ / ] 微调、s 保存）" % args.wrist_offset)
    print("grip 约定: %.0f°（J5=0 张开相对臂前向；压短边差 90° 时 --grip-conv 90/-90）" % args.grip_conv)

    # 眼在手扫描姿态：--scan-pulses 优先，否则读 config.json 的 scan_pulses
    scan_pulses = None
    if args.scan:
        if args.scan_pulses:
            try:
                scan_pulses = [int(x) for x in args.scan_pulses.split(",")]
            except Exception:
                scan_pulses = None
        else:
            _sp = cfg.get("scan_pulses")
            if isinstance(_sp, list) and len(_sp) == 6:
                scan_pulses = [int(p) for p in _sp]
        if scan_pulses is None or len(scan_pulses) != 6:
            print("警告: --scan 需要 6 路脉宽（--scan-pulses 或 config.json 的 scan_pulses），本次跳过扫描姿态")
            scan_pulses = None
        else:
            print("扫描姿态(眼在手): %s（--scan-pulses 或 config.json scan_pulses 覆盖）" % scan_pulses)

    detector = Detector()
    H = load_homography(args.homography)
    ik = ArmIK()
    if args.wrist_ecc is not None:
        ik.wrist_ecc = args.wrist_ecc
    if args.ecc_dir is not None:
        ik.ecc_dir = args.ecc_dir

    host = None
    if args.port and not args.sim:
        tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        from arm_host import ArmHost
        host = ArmHost(serial.Serial(args.port, args.baud, timeout=0.5))
        host.home()
        open_pulses = ik.to_pulses([0, 0, 0, 0, 0, args.grip_open])
        print("初始化：HOME 完成，夹爪张开 ->", host.move(5, open_pulses[5]))

    cap, still = open_camera(args.source)
    next_idx = 0       # 下一个选择的目标序号（循环轮换）
    step_idx = None    # 当前抓取已执行到的步数；None=空闲
    cur_pulses = None  # 当前目标的四段脉宽
    last_hold = None   # 上次抓取的四段脉宽：放物步骤时先降到位、再原地张开、再空爪抬起
    mode = "grab"      # grab=抓取流程 / release=抓完待放物（按空格=降到位+放开+抬起一步）
    auto = False          # 自动连续抓取进行中
    auto_phase = None     # 自动模式阶段: "grab"=抓取中 / "release"=放物
    auto_list = []        # 目标快照 [{"x","y","color","shape"}...]
    auto_i = 0            # 当前目标索引
    auto_t_next = 0.0     # 下一次自动动作的时间戳
    auto_snap_done = False  # 是否已对画面目标做快照
    filter_on = True      # 颜色过滤开关：False 时抓取所有目标（按 c 切换）
    smoother = _EmaSmoother(0.4)   # 帧间平滑：抑制检测框/辅助线的像素级抖动

    # 模式4 鼠标点击：左键点目标抓取，抓完左键点工作区选投放点
    click_phase = "pick" if args.mode == 4 else None
    click_seq = None        # 当前执行序列（4 段抓取 / 3 段放物）
    click_i = 0
    click_t_next = 0.0
    click_sel = None        # 选中目标的画面像素（画红圈）
    click_drop = None       # 投放点的画面像素（画蓝圈）
    cur_wrist = 0.0         # 本次抓取的 J5 腕向角，放物时保持同一朝向
    click_queue = []        # 鼠标左键点击队列（窗口坐标）
    last_lb = (1.0, 0, 0)   # 最近一次 letterbox 的 (scale,left,top)，点击坐标反算用

    def release_here():
        """上一轮抓完还抓着物体时：放物一步（降到位+放开+抬起），避免带物移动甩出。"""
        nonlocal last_hold
        if last_hold is not None:
            release_pulses(host, ik, args, last_hold)
            last_hold = None

    def go_scan():
        """眼在手：移动到扫描姿态并等待画面稳定。仅在 --scan 且脉宽有效时使用。"""
        if host is None:
            print("  [sim] 扫描姿态 %s" % scan_pulses)
        else:
            host.joint(*scan_pulses)
            time.sleep(1.0)   # 等机械臂到位、摄像头画面稳定后再识别

    if scan_pulses is not None:
        go_scan()
        print("已到扫描姿态(眼在手)，开始识别")

    # 窗口可缩放；等比约束由主循环实时纠正实现（见主循环 resizeWindow 逻辑）
    cv2.namedWindow("grasp", cv2.WINDOW_NORMAL)
    if args.mode == 4:
        def _on_mouse(ev, mx, my, fl, prm):
            if ev == cv2.EVENT_LBUTTONDOWN:
                click_queue.append((mx, my))
        cv2.setMouseCallback("grasp", _on_mouse)
    prev_win = (0, 0)

    try:
        while True:
            if still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            targets = detector.detect(frame)
            targets = smoother(targets)   # 帧间 EMA 平滑，稳定中心点与方向角
            disp = draw_targets(frame.copy(), targets)

            # 自动模式：第一帧对画面目标做快照并开始依次抓取
            if args.mode == 2 and not auto_snap_done:
                auto_snap_done = True
                matched = [t for t in targets if not (filter_on and args.color) or t.color == args.color]
                if not matched:
                    print("画面中没有可抓目标。按 h 回 HOME 后重试，或按 r 返回选择菜单")
                else:
                    for t in matched:
                        wx, wy = pixel_to_world(H, t.cx, t.cy)
                        x = wx / 10.0 - args.base_x + args.off_x
                        y = wy / 10.0 - args.base_y + args.off_y
                        auto_list.append({
                            "x": x, "y": y,
                            "wrist": calc_wrist(ik, H, t, x, y, args.wrist_offset, args.grip_conv),
                            "color": t.color, "shape": t.shape,
                        })
                    print("自动模式：画面检测到 %d 个目标，开始依次抓取（按 h 停止，r 返回选择菜单，q 退出）"
                          % len(auto_list))
                    # 选第一个可达目标（不可达的跳过，与其他目标一致）
                    cur_pulses = None
                    while auto_i < len(auto_list):
                        sel = auto_list[auto_i]
                        cur_pulses, err, tilt = grab_pulses(ik, args, sel["x"], sel["y"], sel.get("wrist", 0.0))
                        if err:
                            print("  目标 #%d 不可达: %s，跳过" % (auto_i, err))
                            auto_i += 1
                            continue
                        break
                    if cur_pulses is None:
                        print("  全部目标不可达，自动模式未启动。按 r 返回选择菜单后重试")
                    else:
                        sel = auto_list[auto_i]
                        print("\n自动抓取 #%d %s/%s arm(%.1f,%.1f)cm"
                              % (auto_i, sel["color"], sel["shape"], sel["x"], sel["y"]))
                        if tilt > 0:
                            print("  目标较远，采用前倾 %.0f° 抓取以扩大臂展" % tilt)
                        step_idx = 0
                        do_step(host, cur_pulses[step_idx], STEP_NAMES[step_idx])
                        auto = True
                        auto_phase = "grab"
                        auto_t_next = time.time() + args.step_wait

            # 自动模式推进：到点执行下一步动作（抓取 4 步 -> 放物 -> 下一个目标）
            now = time.time()
            if auto and now >= auto_t_next:
                if auto_phase == "grab" and step_idx is not None:
                    step_idx += 1
                    if step_idx >= len(STEP_NAMES):
                        print("  抓取完成，自动放物")
                        last_hold = cur_pulses
                        cur_pulses = None
                        step_idx = None
                        auto_phase = "release"
                        auto_t_next = now
                    else:
                        do_step(host, cur_pulses[step_idx], STEP_NAMES[step_idx])
                        if step_idx >= 2:
                            last_hold = cur_pulses   # 闭合完成即可安全放物：闭合后到抬起前按 h/r 也能先放物
                        auto_t_next = now + args.step_wait
                elif auto_phase == "release":
                    release_pulses(host, ik, args, last_hold)
                    last_hold = None
                    auto_phase = "grab"
                    auto_i += 1
                    while auto_i < len(auto_list):
                        sel = auto_list[auto_i]
                        cur_pulses, err, tilt = grab_pulses(ik, args, sel["x"], sel["y"], sel.get("wrist", 0.0))
                        if err:
                            print("  目标 #%d 不可达: %s，跳过" % (auto_i, err))
                            auto_i += 1
                            continue
                        break
                    if auto_i >= len(auto_list):
                        auto = False
                        auto_phase = None
                        print("自动连续抓取完成（%d 个目标全部处理）。按空格重新开始自动抓取（按 r 返回选择菜单）"
                              % len(auto_list))
                    else:
                        sel = auto_list[auto_i]
                        print("\n自动抓取 #%d %s/%s arm(%.1f,%.1f)cm"
                              % (auto_i, sel["color"], sel["shape"], sel["x"], sel["y"]))
                        if tilt > 0:
                            print("  目标较远，采用前倾 %.0f° 抓取以扩大臂展" % tilt)
                        step_idx = 0
                        do_step(host, cur_pulses[step_idx], STEP_NAMES[step_idx])
                        auto_t_next = time.time() + args.step_wait

            # 模式4 鼠标点击：序列推进（抓取 4 步 / 放物 3 步）+ 点击事件处理
            if args.mode == 4:
                now4 = time.time()
                if click_phase in ("grab", "dropping") and click_seq is not None and now4 >= click_t_next:
                    do_step(host, click_seq[click_i],
                            STEP_NAMES[click_i] if click_phase == "grab" else DROP_NAMES[click_i])
                    click_i += 1
                    if click_phase == "grab" and click_i >= len(STEP_NAMES):
                        # 抓取完成：停在原位夹稳，等点击投放点（不回 HOME）
                        wait_auto(host, max(args.step_wait, args.click_wait))
                        print("  抓取完成，停在原位夹稳；左键点击工作区任意点选择投放位置")
                        last_hold = click_seq
                        cur_pulses = None
                        click_phase = "drop"
                        mode = "release"
                        click_sel = None
                        click_seq = None
                    elif click_phase == "dropping" and click_i >= len(DROP_NAMES):
                        # 放物完成：等空爪抬起到位后再回 HOME，可开始新一轮
                        wait_auto(host, max(args.step_wait, args.click_wait))
                        if host is not None:
                            host.home()
                            print("  放物完成，回 HOME …")
                            wait_auto(host, max(args.step_wait, args.click_wait))
                            print("  已回 HOME，左键点击目标开始新一轮抓取")
                        else:
                            print("  [sim] 放物完成，回 HOME")
                        click_phase = "pick"
                        mode = "grab"
                        click_drop = None
                        click_seq = None
                    else:
                        click_t_next = now4 + args.click_wait
                while click_queue:
                    mx, my = click_queue.pop(0)
                    _sc, _lx, _ty = last_lb
                    fx = (mx - _lx) / _sc if _sc > 0 else mx
                    fy = (my - _ty) / _sc if _sc > 0 else my
                    if click_phase in ("grab", "dropping"):
                        print("  动作执行中，请稍候…")
                        continue
                    if click_phase == "drop":
                        wx, wy = pixel_to_world(H, fx, fy)
                        x, y = wx / 10.0 - args.base_x, wy / 10.0 - args.base_y
                        seq, err, tilt = drop_pulses(ik, args, x, y, cur_wrist)
                        if err:
                            print("  投放点不可达: %s" % err)
                            continue
                        if tilt > 0:
                            print("  投放点较远，采用前倾 %.0f° 放物" % tilt)
                        print("  投放点 arm(%.1f,%.1f)cm，开始放物" % (x, y))
                        click_seq = seq
                        click_i = 0
                        click_t_next = time.time()
                        click_phase = "dropping"
                        click_drop = (int(fx), int(fy))
                        continue
                    if not targets:
                        print("  画面中没有目标（可能被机械臂遮挡），按 h 回 HOME 后重试")
                        continue
                    matched = [t for t in targets if not (filter_on and args.color) or t.color == args.color]
                    if not matched:
                        print("  没有可抓的目标（颜色过滤: %s）" % args.color)
                        continue
                    best = min(matched, key=lambda t: (t.cx - fx) ** 2 + (t.cy - fy) ** 2)
                    # 必须点在目标包围框内（宽/高一半再外扩 15px）才算选中，点空白不触发抓取
                    if not (abs(fx - best.cx) <= best.w / 2.0 + 15.0 and abs(fy - best.cy) <= best.h / 2.0 + 15.0):
                        print("  点击处不在目标上（请点击画面中的目标）")
                        continue
                    wx, wy = pixel_to_world(H, best.cx, best.cy)
                    x, y = wx / 10.0 - args.base_x + args.off_x, wy / 10.0 - args.base_y + args.off_y
                    wrist = calc_wrist(ik, H, best, x, y, args.wrist_offset, args.grip_conv)
                    cur_wrist = wrist
                    print("\n抓取 %s/%s  px(%.0f,%.0f) mm(%.1f,%.1f) arm(%.1f,%.1f)cm"
                          % (best.color, best.shape, best.cx, best.cy, wx, wy, x, y))
                    cur_pulses, err, tilt = grab_pulses(ik, args, x, y, wrist)
                    if err:
                        print("  ->", err)
                        continue
                    if tilt > 0:
                        print("  目标较远，采用前倾 %.0f° 抓取以扩大臂展" % tilt)
                    click_seq = cur_pulses
                    click_i = 0
                    click_t_next = time.time()
                    click_phase = "grab"
                    click_sel = (int(best.cx), int(best.cy))

            for i, t in enumerate(targets):
                wx, wy = pixel_to_world(H, t.cx, t.cy)
                x, y = wx / 10.0 - args.base_x, wy / 10.0 - args.base_y
                lines = ["#%d %s/%s" % (i, t.color, t.shape), "mm(%.0f,%.0f)" % (wx, wy)]
                for dy, text in ((60, lines[0]), (74, lines[1])):
                    cv2.putText(disp, text, (int(t.cx) - 50, int(t.cy) + dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # 夹爪张开方向辅助线（绿色）：两指连线方向，与长边（红线）垂直为正确。
                # 由 calc_wrist 实时算出（含 wrist-offset），调 [ / ] 时绿线会转动。
                if t.shape in ("square", "rectangle", "diamond"):
                    xg, yg = x + args.off_x, y + args.off_y
                    wrist = calc_wrist(ik, H, t, xg, yg, args.wrist_offset, args.grip_conv)
                    open_ang = math.radians(_axis_th1(ik, xg, yg, wrist) + 90.0 + args.grip_conv + wrist)   # 张开方向(世界角)
                    w0 = pixel_to_world(H, t.cx, t.cy)
                    w1 = (w0[0] + 10.0 * math.cos(open_ang), w0[1] + 10.0 * math.sin(open_ang))
                    p1 = world_to_pixel(H, w1[0], w1[1])
                    dx, dy = p1[0] - t.cx, p1[1] - t.cy
                    L = math.hypot(dx, dy) or 1.0
                    half = min(t.w, t.h) / 2.0      # 绿线长度 = 物体短边半宽，跨过宽度
                    ux, uy = dx / L, dy / L
                    cv2.line(disp, (int(t.cx), int(t.cy)),
                             (int(t.cx + ux * half), int(t.cy + uy * half)), (0, 255, 0), 2)
                    cv2.line(disp, (int(t.cx), int(t.cy)),
                             (int(t.cx - ux * half), int(t.cy - uy * half)), (0, 255, 0), 2)

            fh = disp.shape[0]
            if args.roi is not None:
                rx, ry, rw, rh = args.roi
                draw_dashed_rect(disp, rx, ry, rw, rh, (0, 255, 255), dash=10, gap=6, thickness=2)
                cv2.putText(disp, "ROI", (rx + 3, max(12, ry - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(disp, "space: step   [:w-2   ]:w+2   s:save   c: filter   h: home   r: menu   q: quit", (10, fh - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if args.mode == 4:
                if click_phase == "pick":
                    status = "click4: 左键点目标抓取  wrist-offset=%.1f°" % args.wrist_offset
                elif click_phase == "grab":
                    status = "click4: 抓取中 %d/4 %s" % (min(click_i + 1, 4), STEP_NAMES[min(click_i, 3)])
                elif click_phase == "drop":
                    status = "click4: 左键点工作区选投放点"
                else:
                    status = "click4: 放物中 %d/3 %s" % (min(click_i + 1, 3), DROP_NAMES[min(click_i, 2)])
                if filter_on and args.color:
                    status += "  filter: %s" % args.color
            elif mode == "release":
                status = "grab: 待放物 (space = 降到位+放开+抬起)"
            elif auto:
                status = "AUTO: %d/%d %s" % (
                    auto_i + 1, len(auto_list),
                    "放物" if auto_phase == "release" else
                    ("步 %d/4 %s" % (step_idx + 1, STEP_NAMES[step_idx]) if step_idx is not None else "等待"))
            elif step_idx is None:
                status = "grab: idle (space selects next target)  wrist-offset=%.1f°" % args.wrist_offset
                if filter_on and args.color:
                    status += "  filter: %s" % args.color
            else:
                status = "grab: %d/%d %s" % (step_idx + 1, len(STEP_NAMES), STEP_NAMES[step_idx])
            cv2.putText(disp, status, (10, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if args.mode == 4:
                if click_sel is not None:
                    cv2.circle(disp, click_sel, 28, (0, 0, 255), 2)
                    cv2.circle(disp, click_sel, 3, (0, 0, 255), -1)
                if click_drop is not None:
                    cv2.circle(disp, click_drop, 28, (255, 0, 0), 2)
                    cv2.circle(disp, click_drop, 3, (255, 0, 0), -1)
            # 拖动过程中强制窗口长宽等比：以用户拖动主导的边为基准，实时纠正另一条边
            try:
                rect = cv2.getWindowImageRect("grasp")
            except cv2.error:
                # 窗口被点 X 关闭：收尾（放物、回 HOME）后退出，避免窗口句柄失效崩溃
                release_here()
                if host is not None:
                    host.home()
                    print("窗口已关闭，已回 HOME，程序退出")
                else:
                    print("  [sim] 窗口已关闭，程序退出")
                break
            if rect[2] > 0 and rect[3] > 0:
                fw, fh = disp.shape[1], disp.shape[0]
                cw, ch = rect[2], rect[3]
                if abs(cw * fh - ch * fw) > max(4, int(0.02 * max(cw, ch))):
                    dw, dh = cw - prev_win[0], ch - prev_win[1]
                    if abs(dw) >= abs(dh):
                        cv2.resizeWindow("grasp", cw, max(1, int(round(cw * fh / fw))))
                    else:
                        cv2.resizeWindow("grasp", max(1, int(round(ch * fw / fh))), ch)
                    rect = cv2.getWindowImageRect("grasp")
                    cw, ch = rect[2], rect[3]
                prev_win = (cw, ch)
            if rect[2] > 0 and rect[3] > 0 and (rect[2] != disp.shape[1] or rect[3] != disp.shape[0]):
                _fw0, _fh0 = disp.shape[1], disp.shape[0]
                _sc = min(rect[2] / _fw0, rect[3] / _fh0)
                _nw, _nh = max(1, int(round(_fw0 * _sc))), max(1, int(round(_fh0 * _sc)))
                last_lb = (_sc, (rect[2] - _nw) // 2, (rect[3] - _nh) // 2)
                disp = letterbox(disp, rect[2], rect[3])
            cv2.imshow("grasp", disp)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                # 退出前安全收尾：若有物体在爪里先放物，再回 HOME 恢复视野，避免姿态异常
                release_here()
                if host is not None:
                    host.home()
                    print("已回 HOME，程序退出")
                else:
                    print("  [sim] 回 HOME")
                break
            if key == ord("h"):
                # 回 HOME：若还抓着物体，先放物，避免带物回 HOME 撞到别处
                release_here()
                if host is not None:
                    host.home()
                    print("已回 HOME，视野恢复")
                else:
                    print("  [sim] 回 HOME")
                step_idx = None
                cur_pulses = None
                mode = "grab"
                auto = False          # 停止自动模式（不自动重启，按空格重新开始或按 r 返回菜单）
                auto_phase = None
                click_phase = "pick" if args.mode == 4 else None
                click_seq = None
                click_i = 0
                click_sel = None
                click_drop = None
                continue
            if key == ord("r"):
                # 返回选择菜单：先安全放物（若有物体在爪里），回 HOME 恢复视野，
                # 再重新选模式并重置所有状态，避免机械臂停在半空姿态进入新模式
                release_here()
                if host is not None:
                    host.home()
                    print("已回 HOME，视野恢复")
                else:
                    print("  [sim] 回 HOME")
                new_mode = choose_mode()
                if new_mode == 3:
                    if not args.color:
                        args.color = choose_color()
                    args.mode = 1   # 模式3 = 指定颜色后按逐步抓取
                else:
                    args.mode = new_mode
                step_idx = None
                cur_pulses = None
                last_hold = None
                mode = "grab"
                next_idx = 0
                auto = False
                auto_phase = None
                auto_list = []
                auto_i = 0
                auto_t_next = 0.0
                auto_snap_done = False
                filter_on = True   # 新模式默认恢复颜色过滤（若指定过颜色）
                click_phase = "pick" if args.mode == 4 else None
                click_seq = None
                click_i = 0
                click_sel = None
                click_drop = None
                print("已切换到模式 %d：%s%s" % (
                    args.mode,
                    "自动连续抓取" if args.mode == 2 else ("鼠标点击模式" if args.mode == 4 else "逐步抓取"),
                    "（只抓 %s）" % args.color if args.color else ""))
                if scan_pulses is not None:
                    go_scan()
                    print("已回扫描姿态，按空格选择目标/开始")
                continue
            if key == ord("c"):
                # 切换颜色过滤：--color/模式3 指定颜色后，模式1/2 中按 c 取消/恢复过滤
                if args.color is None:
                    print("当前未指定颜色（--color/模式3），无颜色过滤可切换")
                    continue
                filter_on = not filter_on
                if filter_on:
                    print("颜色过滤已开启：只抓 %s（按 c 取消）" % args.color)
                else:
                    print("颜色过滤已取消：抓取所有目标（按 c 恢复）")
                if auto:
                    print("  （自动模式中，过滤切换将在下次快照/重新开始时生效）")
                continue
            if key == ord("["):
                args.wrist_offset -= 2.0
                print("wrist offset -> %.1f°（下次抓取生效；按 s 保存到 config.json）" % args.wrist_offset)
                continue
            if key == ord("]"):
                args.wrist_offset += 2.0
                print("wrist offset -> %.1f°（下次抓取生效；按 s 保存到 config.json）" % args.wrist_offset)
                continue
            if key == ord("s"):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        _cfg = json.load(f)
                except Exception:
                    _cfg = {}
                _cfg["wrist_offset"] = args.wrist_offset
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(_cfg, f, indent=2, ensure_ascii=False)
                print("已保存 wrist_offset = %.1f° 到 config.json" % args.wrist_offset)
                continue
            if key == ord(" "):
                if args.mode == 4:
                    print("鼠标点击模式：左键点目标抓取，抓完左键点工作区选投放点（空格无操作）")
                    continue
                if auto:
                    print("自动模式进行中（按 h 停止，r 返回选择菜单，q 退出）")
                    continue
                if mode == "release":
                    # 放物一步：降到位 -> 原地张开 -> 空爪抬起（内部自动等待舵机/落稳）
                    release_here()
                    mode = "grab"
                    print("  已放下物体，按空格选择下一个目标（按 r 返回选择菜单）")
                    continue
                if step_idx is not None:
                    # 进行中：执行下一步，不依赖当前帧是否还能看到目标
                    # （机械臂到位后通常会遮挡摄像头里的目标）
                    step_idx += 1
                    if step_idx >= len(STEP_NAMES):
                        print("  本次抓取完成，按空格放下物体（降到位+放开+抬起一步完成）")
                        last_hold = cur_pulses   # 记住四段脉宽，放物步骤使用
                        step_idx = None
                        cur_pulses = None
                        mode = "release"
                        continue
                    do_step(host, cur_pulses[step_idx], STEP_NAMES[step_idx])
                    if step_idx >= 2:
                        last_hold = cur_pulses   # 闭合完成即可安全放物：闭合后到抬起前按 h/r 也能先放物
                    continue
                # 眼在手：空闲时先回扫描姿态，获取新的正上方画面再选目标/重启自动
                if scan_pulses is not None:
                    go_scan()
                    print("已回扫描姿态，再按一次空格选择目标/开始")
                    continue
                if args.mode == 2:
                    # 自动模式空闲（已结束/未开始）：按空格重新开始自动抓取，不进入手动模式
                    # 放在无目标判断之前：全部抓完后画面常无目标，空格仍应重启（快照空则提示）
                    auto_list = []
                    auto_i = 0
                    auto_snap_done = False
                    print("重新开始自动连续抓取…")
                    continue
                if not targets:
                    print("画面中没有目标（可能被机械臂遮挡）。按 h 回 HOME 移开后重试，或按 r 返回选择菜单")
                    continue
                matched = [t for t in targets if not (filter_on and args.color) or t.color == args.color]
                if not matched:
                    print("没有可抓的目标（颜色过滤: %s）" % args.color)
                    continue
                # 空闲：选目标，立即执行第 1 步（准备）
                sel = matched[next_idx % len(matched)]
                next_idx += 1
                wx, wy = pixel_to_world(H, sel.cx, sel.cy)
                x, y = wx / 10.0 - args.base_x + args.off_x, wy / 10.0 - args.base_y + args.off_y
                wrist = calc_wrist(ik, H, sel, x, y, args.wrist_offset, args.grip_conv)
                print("\n抓取 #%d %s/%s  px(%.0f,%.0f) mm(%.1f,%.1f) arm(%.1f,%.1f)cm%s%s"
                      % (targets.index(sel), sel.color, sel.shape, sel.cx, sel.cy, wx, wy, x, y,
                         ("  偏移x%.1f y%.1f z%.1f" % (args.off_x, args.off_y, args.off_z))
                         if (args.off_x or args.off_y or args.off_z) else "",
                         "  wrist=%.0f°(沿边)" % wrist if wrist else ""))
                cur_pulses, err, tilt = grab_pulses(ik, args, x, y, wrist)
                if err:
                    print("  ->", err)
                    continue
                if tilt > 0:
                    print("  目标较远，采用前倾 %.0f° 抓取以扩大臂展" % tilt)
                step_idx = 0
                do_step(host, cur_pulses[step_idx], STEP_NAMES[step_idx])
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if host is not None:
            host.close()


if __name__ == "__main__":
    main()
