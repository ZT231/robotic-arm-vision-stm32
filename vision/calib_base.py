#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定 J1 转轴在标定坐标系中的位置（base_x/base_y）。

homography 四点法把工作台映射成毫米坐标系（原点=①，X 右，Y 前）。
机械臂坐标 = 视觉毫米/10 - base（cm），其中 base 就是 J1 转轴投影点在
标定坐标系里的坐标。本工具：打开摄像头画面，点击 J1 舵盘中心铅垂投影
点（可多点取平均），自动换算并写入 config.json。

用法（在 vision 目录下）：
  python calib_base.py --source 0 --homography homography.json
  python calib_base.py --ref --source 0 --homography homography.json   # J1 投影不在画面内时
  python calib_base.py --source worktable.png   # 或对静态图点击

按键：
  鼠标左键  添加一个采样点（J1 投影点；--ref 时为画面内标记物）
  c         清除所有点
  空格/回车 确认并写入 config.json（取平均）
  q         退出不保存

--ref 参考点模式（J1 投影点不在画面内时用）：
  点击画面内一个台面标记物（如硬币/笔尖/夹爪夹的小方块），按空格后
  在终端输入它相对 J1 投影点的偏移 dx,dy（cm，臂右为正 X、臂前为正 Y），
  工具自动反推 base：base = 视觉毫米/10 - 偏移。

注意：
  - 点击点必须与当前抓取用的 homography 是同一套标定。
  - 若 J1 投影点不在画面内，改用夹爪夹方块移到画面内，
    以方块中心相对①的偏移填 base（符号：臂右为正 X、臂前为正 Y）。
"""

import argparse
import json
import os
import time

import cv2

from homography import load_homography, pixel_to_world

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _open(source):
    if os.path.splitext(str(source))[1].lower() in IMG_EXTS:
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"无法读取图片 {source}")
        return None, img
    idx = int(source) if str(source).isdigit() else source
    if isinstance(idx, int):
        # Windows 用 DSHOW 后端，避免 MSMF 的 read() 卡死
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise ValueError(f"无法打开摄像头 {source}")
    for _ in range(8):
        ok, _ = cap.read()
        if ok:
            return cap, None
        time.sleep(0.2)
    cap.release()
    raise ValueError(f"摄像头 {source} 无法读取画面（请检查是否被其他程序占用）")


def main():
    ap = argparse.ArgumentParser(description="标定 J1 转轴位置 base_x/base_y")
    ap.add_argument("--source", default="0", help="摄像头编号或图片路径")
    ap.add_argument("--homography", default="homography.json", help="单应矩阵文件")
    ap.add_argument("--config", default=None, help="config.json 路径，缺省为本目录 config.json")
    ap.add_argument("--ref", action="store_true",
                    help="参考点模式：J1 投影不在画面内时，点画面内标记物+输入其相对 J1 的偏移")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    H = load_homography(args.homography)

    cap, still = _open(args.source)
    if still is not None:
        frame = still.copy()
    else:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("无法从摄像头读取画面")
    cap.release()
    frame = frame.copy()

    pts = []   # 点击的像素点

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow("calib-base")
    cv2.setMouseCallback("calib-base", on_click)
    while True:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if pts:
            wx = sum(pixel_to_world(H, x, y)[0] for x, y in pts) / len(pts)
            wy = sum(pixel_to_world(H, x, y)[1] for x, y in pts) / len(pts)
            if args.ref:
                line = "标记物 mm=(%.1f, %.1f)   n=%d   （空格后输入相对 J1 的偏移）" % (wx, wy, len(pts))
            else:
                line = "base = (%.2f, %.2f) cm   mm=(%.1f, %.1f)   n=%d" % (
                    wx / 10.0, wy / 10.0, wx, wy, len(pts))
        else:
            line = ("点击画面内标记物（多点取平均）" if args.ref
                    else "点击 J1 舵盘中心正下方台面的投影点（多点取平均）")
        cv2.putText(disp, line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, "L:add  c:clear  space/enter:save  q:quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calib-base", disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            pts.clear()
        if key in (ord(" "), 13) and pts:
            wx = sum(pixel_to_world(H, x, y)[0] for x, y in pts) / len(pts)
            wy = sum(pixel_to_world(H, x, y)[1] for x, y in pts) / len(pts)
            if args.ref:
                s = input("标记物相对 J1 投影的偏移 dx,dy (cm, 臂右为正X 臂前为正Y): ").strip()
                try:
                    dx, dy = (float(v) for v in s.replace("，", ",").split(","))
                except Exception:
                    print("输入格式应为 dx,dy，如 5,8；本次不保存")
                    continue
                bx, by = wx / 10.0 - dx, wy / 10.0 - dy
            else:
                bx, by = wx / 10.0, wy / 10.0
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
            cfg["base_x"] = round(bx, 2)
            cfg["base_y"] = round(by, 2)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print("已保存 base_x=%.2f base_y=%.2f (cm) -> %s" % (bx, by, cfg_path))
            print("验证：抓取时把目标放在 J1 正上方，画面读数 arm 坐标应约 (0,0)")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
