#!/usr/bin/env python3
"""S5 串口联调探针：验证固件通信与新 J3/J4 软限位。

用法（在 vision 目录下）：
  python arm_probe.py --port COM3
每步按回车确认，机械臂动作前请确保周围无遮挡。
"""

import argparse
import os
import sys

vision_dir = os.path.dirname(os.path.abspath(__file__))
tools = os.path.join(vision_dir, "..", "tools")
if tools not in sys.path:
    sys.path.insert(0, tools)

from arm_host import ArmHost

ap = argparse.ArgumentParser(description="S5 串口联调探针")
ap.add_argument("--port", required=True, help="串口，如 COM3")
ap.add_argument("--baud", type=int, default=115200)
args = ap.parse_args()

host = ArmHost(args.port, args.baud)
host.log(0)
print("连接 %s 成功\n" % args.port)

print("[1] HOME（回中位）...")
host.home()
input("    观察是否回到中位姿态，回车继续\n")

print("[2] J3 -> 2000us（应在旧限位内，小臂前倾约 70°）")
print("    响应:", host.move(2, 2000))
input("    回车继续\n")

print("[3] J3 -> 2400us（新限位 2500 内，小臂前倾约 118°）")
print("    响应:", host.move(2, 2400))
input("    确认小臂能继续前倾（旧固件会被裁剪到 2150，转不过去），回车继续\n")

print("[4] J4 -> 2400us（新限位 2500 内，腕关节大幅前倾）")
print("    响应:", host.move(3, 2400))
input("    回车继续\n")

print("[5] 回 HOME")
host.home()
host.close()
print("完成，串口已关闭")
