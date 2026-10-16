#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5 关节角度标定引导：逐关节量角，生成 [pulse, deg] 映射表，写回 arm_params.json。

原理：舵机角度与脉宽近似线性，但大角度区有明显非线性。对每个关节在
min/25%/中位/75%/max 五个脉宽点量出实际角度（相对角，见 ik.py 顶部约定），
生成 map = [[pulse, deg], ...]；ik.py 用分段线性插值消除非线性误差，
同时保留线性拟合的 deg_min/deg_max 作为无 map 时的兜底。

量角基准（每关节不同，提示里会说明；与 ik.py 角度约定一致，前倾为负、后仰为正）：
  J1 基座方位角：俯视，相对正前方，右(顺)为负，左(逆)为正
  J2 大臂仰角：  侧视，竖直向上=0°，前伸(顺)为负，后伸(逆)为正
  J3 小臂相对大臂：侧视，小臂沿大臂延伸方向=0°，前伸为负，后伸为正（标定时大臂在中位）
  J4 夹爪相对小臂：侧视，夹爪沿小臂延伸方向=0°，前伸为负，后伸为正（标定时小臂在中位）
  J5 腕旋转：    俯视，相对正前方，右(顺)为负，左(逆)为正
  J6 夹爪开合：  0=闭合，正=张开，90=张开最大

量角工具：手机量角器 App（推荐）或拍照后画图工具量角。

用法：
  python calib_joints.py --sim                       # 无硬件模拟，熟悉流程
  python calib_joints.py --port COM3                 # 真机标定 J1~J4
  python calib_joints.py --port COM3 --joints 1,2,3,4,5,6
  python calib_joints.py --port COM3 --apply         # 标定完直接写回 arm_params.json

按键流程：每个脉宽点转到位后，输入量得的角度(回车)，自动进入下一点。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
MID_PULSE = 1500
JOINT_BASE = {
    "J1": "俯视：基座指向 相对 机械臂正前方（右顺为负，左逆为正）",
    "J2": "侧视：大臂轴线 相对 竖直向上（竖直向上=0°，前伸为负，后伸为正）",
    "J3": "侧视：小臂轴线 相对 大臂延伸方向（沿大臂=0°，前伸为负，后伸为正；J2 已在中位）",
    "J4": "侧视：夹爪段 相对 小臂延伸方向（沿小臂=0°，前伸为负，后伸为正；J3 已在中位）",
    "J5": "俯视：夹爪朝向 相对 正前方（右顺为负，左逆为正）",
    "J6": "正视：夹爪开合角（-40=完全闭合，0=适中，+50=张开最大，实测）",
}


def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_params(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print("已写入 %s" % os.path.abspath(path))


def ask_angle(joint_name, pulse):
    while True:
        try:
            s = input("  %s @ %dus，请量角并输入（度，可带负号/小数）: " % (joint_name, pulse)).strip()
            if s:
                return float(s)
        except (EOFError, KeyboardInterrupt):
            raise
        except ValueError:
            print("  输入无效，重新输入")


def wait_settle(host, ch, target, timeout=8.0):
    """轮询 STATE，等该通道 current 到达 target（软件平滑到位）。返回到位时脉宽，超时返回 None。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            _, pulses = host.state_pulses()
        except Exception:
            time.sleep(0.2)
            continue
        if abs(pulses[ch] - target) <= 20:
            return pulses[ch]
        time.sleep(0.2)
    return None


def calibrate_joint(host, ik, joint_idx, apply, out, sim):
    name = JOINT_NAMES[joint_idx]
    j = ik.joints[name]
    mid = int(j.get("pulse_mid", MID_PULSE))
    points = [
        j["pulse_min"],
        int(j["pulse_min"] + (mid - j["pulse_min"]) * 0.5),
        mid,
        int(mid + (j["pulse_max"] - mid) * 0.5),
        j["pulse_max"],
    ]

    # 先把其他关节移到实测中位（pulse_mid），目标关节保持不动
    others_mid = [int(ik.joints[n].get("pulse_mid", MID_PULSE)) for n in JOINT_NAMES]
    for ch, us in enumerate(others_mid):
        if ch != joint_idx:
            if host:
                print("  移动 %s 到中位 %d..." % (JOINT_NAMES[ch], us))
                host.move(ch, us)
                wait_settle(host, ch, us)
                time.sleep(0.3)
    time.sleep(1.0)

    print("\n== 标定 %s ==" % name)
    print("   基准：%s" % JOINT_BASE[name])
    print("   测试脉宽点：%s" % points)
    print("   （若某端舵机已到机械限位卡住，就量它实际能到的最大角度）")
    print("   !!! 限位点（第 1 个和第 5 个点）舵机会顶死堵转、持续大电流，请务必快速量角（10 秒内）！")
    angles = []
    for i, pulse in enumerate(points):
        if i in (0, len(points) - 1):
            print("   >>> %dus 是限位点：舵机正在堵转，立刻量角输入，输完马上卸载。" % pulse)
        if host:
            r = host.move(joint_idx, pulse)
            print("  移动 %s 到 %dus -> %r" % (name, pulse, r))
            if host.is_ok(r):
                cur = wait_settle(host, joint_idx, pulse)
                if cur is None:
                    print("  [警告] %s 未到位（目标 %dus）：舵机可能卡死/堵转，请检查后继续" % (name, pulse))
                else:
                    print("  到位：%s current=%dus" % (name, cur))
            time.sleep(0.5)
        else:
            print("  [sim] %s 应转到 %dus" % (name, pulse))
        angles.append(ask_angle(name, pulse))

    # 线性拟合 deg = k*pulse + b（作为无 map 时的兜底）
    k, b = np.polyfit(points, angles, 1)
    deg_min = k * j["pulse_min"] + b
    deg_max = k * j["pulse_max"] + b
    deg_mid = k * mid + b
    fitted = [k * p + b for p in points]
    err = max(abs(f - a) for f, a in zip(fitted, angles))

    print("  %s: 量得 %s -> 拟合 deg_min=%.1f deg_max=%.1f (线性误差 %.1f°)" %
          (name, ["%.1f" % a for a in angles], deg_min, deg_max, err))

    result = {
        "pulse_min": j["pulse_min"],
        "pulse_max": j["pulse_max"],
        "measured_angles": angles,
        "map": [[int(p), round(a, 1)] for p, a in zip(points, angles)],
        "deg_min": round(deg_min, 1),
        "deg_max": round(deg_max, 1),
        "linear_error_deg": round(err, 1),
    }
    out[name] = result

    if apply:
        j["deg_min"] = round(deg_min, 1)
        j["deg_max"] = round(deg_max, 1)
        j["map"] = result["map"]
        print("  -> 已更新 %s.deg_min/deg_max/map" % name)
    return result


def main():
    ap = argparse.ArgumentParser(description="S5 关节角度范围标定引导")
    ap.add_argument("--port", default=None, help="串口（如 COM3），缺省为 sim 模拟")
    ap.add_argument("--sim", action="store_true", help="模拟模式（不连串口，仅演示流程）")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--joints", default="1,2,3,4", help="要标定的关节，逗号分隔，默认 1,2,3,4")
    ap.add_argument("--apply", action="store_true", help="标定完直接写回 arm_params.json")
    ap.add_argument("--out", default="joints_calib.json", help="标定结果输出文件")
    args = ap.parse_args()

    vision_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(vision_dir, "arm_params.json")
    params = load_params(params_path)
    from ik import ArmIK
    ik = ArmIK(params_path)

    host = None
    if args.port and not args.sim:
        tools = os.path.join(vision_dir, "..", "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        from arm_host import ArmHost
        host = ArmHost(args.port, args.baud)
        host.log(0)
        host.home()
        print("已连接 %s，HOME 完成。开始前请确认：" % args.port)
        print("  - 机械臂周围无遮挡、夹爪张开、人员与设备安全")
        print("  - 舵机建议独立 5V/3A+ 电源供电，勿用电脑 USB 直供（限位堵转电流可达 1~2A/个）")

    out = {}
    joints = [int(s.strip()) - 1 for s in args.joints.split(",") if s.strip()]
    joints = [i for i in joints if 0 <= i < 6]
    if not joints:
        print("无效关节编号: %s" % args.joints)
        return

    def save_progress():
        with open(os.path.join(vision_dir, args.out), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if args.apply and out:
            params["joints"] = ik.joints  # calibrate_joint 更新的是 ik.joints，与 params 是两个对象
            save_params(params, params_path)

    try:
        for n, idx in enumerate(joints):
            calibrate_joint(host, ik, idx, args.apply, out, args.port is None)
            save_progress()  # 每标完一个关节立即落盘，中途 Ctrl+C 不丢数据
            if n < len(joints) - 1:
                print("  关节间冷却 2 秒，让舵机休息…")
                time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n已中断。已完成关节的数据已保存。")
        print("  继续时建议只标剩余关节，例如：python calib_joints.py --port COM7 --joints 2,3,4 --apply")
    finally:
        if host:
            host.close()
            print("串口已关闭。")

    print("\n标定结果已保存: %s" % os.path.join(vision_dir, args.out))
    if args.apply and out:
        print("arm_params.json 已更新。可以用 ik.py 自检验证。")


if __name__ == "__main__":
    main()
