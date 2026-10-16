#!/usr/bin/env python3
"""S5 IK 姿态驱动验证：解算目标点 -> 下发真机，观察姿态与夹爪朝向。

用法（在 vision 目录下）：
  python ik_drive.py --port COM7
每步按回车移动到下一点。重点观察：
  1) 夹爪是否竖直朝下（J4 补偿生效）
  2) 机械臂整体是否指向目标方向
  3) 姿态是否舒展、无关节卡顿
"""

import argparse
import os
import sys

vision_dir = os.path.dirname(os.path.abspath(__file__))
tools = os.path.join(vision_dir, "..", "tools")
if tools not in sys.path:
    sys.path.insert(0, tools)

from arm_host import ArmHost
from ik import ArmIK

ap = argparse.ArgumentParser(description="S5 IK 姿态驱动验证")
ap.add_argument("--port", required=True, help="串口，如 COM7")
ap.add_argument("--baud", type=int, default=115200)
args = ap.parse_args()

host = ArmHost(args.port, args.baud)
host.log(0)
host.home()
print("已连接 %s，HOME 完成\n" % args.port)

ik = ArmIK()
points = [
    (0.0, 15.0, "正前方远处"),
    (0.0, 5.0, "正前方近处"),
    (10.0, 10.0, "右前方"),
    (-10.0, 10.0, "左前方"),
    (12.0, 3.0, "右前近处"),
]

try:
    for x, y, desc in points:
        try:
            angles = ik.solve(x, y)
        except ValueError as e:
            print("(%6.1f,%6.1f) %s: 不可达 - %s" % (x, y, desc, e))
            input("    回车继续")
            continue
        pulses = ik.to_pulses(angles)
        print("(%6.1f,%6.1f) %s" % (x, y, desc))
        print("  角度:", " ".join("%6.1f" % a for a in angles))
        print("  脉宽:", " ".join("%5d" % p for p in pulses))
        host.joint(*pulses)
        input("    观察后回车继续\n")
finally:
    print("回 HOME...")
    host.home()
    host.close()
    print("完成，串口已关闭")
