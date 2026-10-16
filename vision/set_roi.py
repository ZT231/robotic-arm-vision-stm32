#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 homography 自动计算 ROI 并写入 config.json。

工作区四角的毫米坐标是已知的：①(0,0) ②(W,0) ③(W,H) ④(0,H)。
homography 是像素<->毫米的双向映射，把四角反算回像素坐标，
取包围盒 + 四周余量即为 ROI（覆盖整个工作区的检测框）。

用法（vision 目录）：
  python set_roi.py --homography homography.json --width 200 --height 250
  python set_roi.py --homography homography.json --width 200 --height 250 --margin 20

说明：
  - 反算出的四角像素若超出画面（如 640x480），说明工作区超出视野，
    需先调整摄像头/工作区布局再标定。
  - --margin 为四周余量像素，默认 15。
"""

import argparse
import json
import os

from homography import load_homography, world_to_pixel


def main():
    ap = argparse.ArgumentParser(description="用 homography 自动计算 ROI 并写入 config.json")
    ap.add_argument("--homography", default="homography.json", help="单应矩阵文件")
    ap.add_argument("--width", type=float, default=200.0, help="工作区宽（毫米）")
    ap.add_argument("--height", type=float, default=250.0, help="工作区高（毫米）")
    ap.add_argument("--margin", type=float, default=15.0, help="四周余量（像素），默认 15")
    ap.add_argument("--config", default=None, help="config.json 路径，缺省为本目录 config.json")
    args = ap.parse_args()

    H = load_homography(args.homography)
    corners_mm = [(0, 0), (args.width, 0), (args.width, args.height), (0, args.height)]
    names = ["①(0,0)", "②(W,0)", "③(W,H)", "④(0,H)"]
    px = []
    for name, (x, y) in zip(names, corners_mm):
        u, v = world_to_pixel(H, x, y)
        px.append((u, v))
        print("角 %s mm(%.0f,%.0f) -> 像素 (%.0f, %.0f)" % (name, x, y, u, v))

    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    x0 = int(min(xs) - args.margin)
    y0 = int(min(ys) - args.margin)
    x1 = int(max(xs) + args.margin)
    y1 = int(max(ys) + args.margin)
    x0, y0 = max(0, x0), max(0, y0)
    roi = [x0, y0, x1 - x0, y1 - y0]
    print("ROI = %s（覆盖整个工作区 + 余量 %dpx）" % (roi, args.margin))

    # 提示：常见画面分辨率下的越界检查
    for w, h in ((640, 480), (1280, 720), (1920, 1080)):
        if roi[0] + roi[2] > w or roi[1] + roi[3] > h:
            print("警告: ROI 超出 %dx%d 画面，请确认工作区在视野内（摄像头高度/角度/布局）" % (w, h))
            break

    cfg_path = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg["roi"] = roi
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("已写入 config.json 的 roi -> %s" % cfg_path)


if __name__ == "__main__":
    main()
