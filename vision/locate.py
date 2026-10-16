"""S4-06 联调：识别目标 -> 像素坐标 -> 毫米坐标，叠加显示并打印。

用法（在 vision 目录下）：
    python locate.py --source 0 --homography homography.json
    python locate.py --source sample.png --homography homography.json

按键：q 退出，s 保存当前画面到 snapshot_locate.png
"""
import argparse
import os

import cv2

from detector import Detector, draw_targets
from homography import load_homography, pixel_to_world

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def run_once(frame, H, det, show=False):
    targets = det.detect(frame)
    for t in targets:
        t.mm_x, t.mm_y = pixel_to_world(H, t.cx, t.cy)
    draw_targets(frame, targets)
    for t in targets:
        cv2.putText(frame, f"mm({t.mm_x:.0f},{t.mm_y:.0f})",
                    (int(t.cx) - 45, int(t.cy) + int(t.h) // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if show:
        cv2.imshow("locate", frame)
        if targets:
            line = "  ".join(f"{t.color}/{t.shape} mm({t.mm_x:.1f},{t.mm_y:.1f})" for t in targets)
            print(f"\r{line}   ", end="")
    else:
        for t in targets:
            print(f"{t.color}/{t.shape}  px({t.cx:.0f},{t.cy:.0f})  mm({t.mm_x:.1f},{t.mm_y:.1f})")
    return targets


def main():
    ap = argparse.ArgumentParser(description="识别 + 像素->毫米 联调")
    ap.add_argument("--source", default="0")
    ap.add_argument("--homography", default="homography.json")
    args = ap.parse_args()

    H = load_homography(args.homography)
    det = Detector()

    if os.path.splitext(str(args.source))[1].lower() in IMG_EXTS:
        frame = cv2.imread(args.source)
        if frame is None:
            raise ValueError(f"无法读取图片 {args.source}")
        run_once(frame, H, det)
        cv2.imwrite("locate_result.png", frame)
        print("已保存 locate_result.png")
        return

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise ValueError(f"无法打开摄像头 {args.source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        run_once(frame, H, det, show=True)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("snapshot_locate.png", frame)
            print("\n已保存 snapshot_locate.png")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()