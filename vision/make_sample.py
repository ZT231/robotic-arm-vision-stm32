"""生成一张离线合成测试图：不同颜色 + 形状的目标放在纯色背景上，用于无摄像头验证识别链路。

用法：
    python make_sample.py                 # 生成 sample.png
    python make_sample.py --out test.png  # 指定输出路径
"""
import argparse
import os

import cv2
import numpy as np

# BGR 颜色（与 config.json 里的 HSV 阈值对应）
COLORS_BGR = {
    "red": (0, 0, 220),
    "green": (0, 180, 0),
    "blue": (220, 0, 0),
    "yellow": (0, 200, 200),
}


def draw_shape(img, shape, center, size, color_bgr):
    cx, cy = center
    s = size
    if shape == "circle":
        cv2.circle(img, (cx, cy), s // 2, color_bgr, -1)
    elif shape == "square":
        cv2.rectangle(img, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), color_bgr, -1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - s // 2], [cx - s // 2, cy + s // 2], [cx + s // 2, cy + s // 2]], np.int32)
        cv2.fillConvexPoly(img, pts, color_bgr)
    elif shape == "rectangle":
        cv2.rectangle(img, (cx - s, cy - s // 2), (cx + s, cy + s // 2), color_bgr, -1)


def make_sample(path="sample.png", bg=(240, 240, 240)):
    img = np.full((480, 640, 3), bg, np.uint8)
    layout = [
        ("red", "circle", (120, 120), 80),
        ("green", "square", (320, 120), 90),
        ("blue", "triangle", (520, 120), 90),
        ("yellow", "rectangle", (120, 340), 90),
        ("red", "triangle", (320, 340), 90),
        ("blue", "square", (520, 340), 90),
    ]
    for color, shape, center, size in layout:
        draw_shape(img, shape, center, size, COLORS_BGR[color])
    cv2.imwrite(path, img)
    print(f"已生成 {os.path.abspath(path)}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample.png")
    args = ap.parse_args()
    make_sample(args.out)