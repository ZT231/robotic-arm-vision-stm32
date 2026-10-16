"""视觉识别演示入口：统一支持图片 / 视频 / 摄像头三种输入。

用法（在 vision 目录下运行）：
    python demo.py                        # 打开默认摄像头 0
    python demo.py --source 0             # 指定摄像头编号
    python demo.py --source sample.png    # 识别单张图片
    python demo.py --source test.mp4      # 识别视频文件
    python demo.py --source sample.png --config config.json   # 指定阈值配置

按键：
    q 退出；s 保存当前画面；空格 暂停/继续（视频/摄像头）
"""
import argparse
import os
import time

import cv2

from detector import Detector, draw_targets

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(path: str) -> bool:
    return os.path.splitext(str(path))[1].lower() in IMG_EXTS


def frames(args):
    """一个生成器：按输入源持续产出帧。图片会反复产出同一帧。"""
    if is_image(args.source):
        frame = cv2.imread(args.source)
        if frame is None:
            raise ValueError(f"无法读取图片：{args.source}")
        while True:
            yield frame
    else:
        src = int(args.source) if str(args.source).isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError(f"无法打开输入源：{args.source}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def main():
    ap = argparse.ArgumentParser(description="机械臂视觉识别演示（颜色+形状）")
    ap.add_argument("--source", default="0", help="摄像头编号 / 图片路径 / 视频路径，默认 0")
    ap.add_argument("--config", default=None, help="阈值配置文件路径，默认 vision/config.json")
    args = ap.parse_args()

    det = Detector(args.config)
    delay = 0 if is_image(args.source) else 1  # 图片用 0 表示等按键，视频/摄像头用 1ms

    paused = False
    for frame in frames(args):
        if not paused:
            targets = det.detect(frame)
        draw_targets(frame, targets)
        cv2.imshow("vision demo - q:quit s:save space:pause", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, frame)
            print(f"已保存 {os.path.abspath(name)}")
        elif key == ord(" "):
            paused = not paused

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()