"""HSV 阈值标定工具：用滑动条在真实/合成画面里调出某个颜色的上下限，
按 p 把当前阈值打印成可直接粘贴进 config.json 的片段。

用法（在 vision 目录下运行）：
    python hsv_tuner.py --source sample.png    # 图图片上标定
    python hsv_tuner.py --source 0             # 实时摄像头标定（推荐）

按键：
    r / g / b / y  载入红/绿/蓝/黄预设阈值
    f              载入粉色（pink）预设阈值
    p              打印当前 HSV 上下限（JSON 片段）
    s              保存当前画面（原图+mask 拼接）
    q              退出

说明：本工具只调「单个颜色的一段阈值」。红色、粉色因色相跨 0/180 两端，需分两段：
    先调 [0,10] 段，再把 H_lo/H_hi 改到 [170,180] 调第二段，最后手动合并进 config.json
    的 red.ranges / pink.ranges。
"""
import argparse
import os
import time

import cv2
import numpy as np

# trackbar 名字与取值范围
TB = ["H_lo", "H_hi", "S_lo", "S_hi", "V_lo", "V_hi"]
TB_MAX = [179, 179, 255, 255, 255, 255]

# 各颜色预设初始值（单段，与 config.json 保持一致）
PRESETS = {
    ord("r"): ("red", [0, 10, 135, 255, 85, 255]),
    ord("g"): ("green", [35, 85, 95, 255, 80, 255]),
    ord("b"): ("blue", [95, 130, 110, 255, 75, 255]),
    ord("y"): ("yellow", [20, 35, 115, 255, 95, 255]),
    ord("f"): ("pink", [0, 10, 60, 130, 150, 255]),
}


def nothing(_):
    pass


def is_image(path: str) -> bool:
    return os.path.splitext(str(path))[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="摄像头编号 / 图片路径")
    args = ap.parse_args()

    if is_image(args.source):
        base = cv2.imread(args.source)
        if base is None:
            raise ValueError(f"无法读取图片：{args.source}")
        use_camera = False
    else:
        src = int(args.source) if str(args.source).isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头：{args.source}")
        use_camera = True

    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)
    for name, mx, default in zip(TB, TB_MAX, PRESETS[ord("r")][1]):
        cv2.createTrackbar(name, "tuner", default, mx, nothing)

    cur = "red"
    print("按 r/g/b/y/f 切换颜色预设，p 打印阈值，q 退出")

    while True:
        if use_camera:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = base.copy()

        vals = [cv2.getTrackbarPos(n, "tuner") for n in TB]
        low = np.array([vals[0], vals[2], vals[4]], np.uint8)
        high = np.array([vals[1], vals[3], vals[5]], np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)

        # 拼接显示：左边原图，右边 mask
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        disp = np.hstack([frame, mask_bgr])
        cv2.putText(disp, f"color={cur}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("tuner", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in PRESETS:
            cur, preset = PRESETS[key]
            for name, v in zip(TB, preset):
                cv2.setTrackbarPos(name, "tuner", v)
            print(f"载入 {cur} 预设")
        elif key == ord("p"):
            print(f"    \"{cur}\": {{\"ranges\": [[[{vals[0]}, {vals[2]}, {vals[4]}], "
                  f"[{vals[1]}, {vals[3]}, {vals[5]}]]]}}")
        elif key == ord("s"):
            name = f"tuner_{int(time.time())}.png"
            cv2.imwrite(name, disp)
            print(f"已保存 {os.path.abspath(name)}")

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()