"""S4-04 相机内参标定：用棋盘格多视角采样，求解相机矩阵和畸变系数。

产出 camera.json（内参矩阵 mtx + 畸变系数 dist + 图像尺寸 + 重投影误差 rms）。
标定结果可用于校正镜头畸变（cv2.undistort），也可作为面试可讲的「相机标定」证据。

用法（在 vision 目录下）：
    python calibrate_camera.py --source 0 --grid 9x6 --square 25 --out camera.json

按键：
    s / 空格   抓取当前帧作为标定样本（仅在检测到棋盘格时有效）
    c          用已抓样本求解并保存
    q          退出

准备：
    1. 打印一张标准棋盘格（内角点数 = --grid，如 9x6），贴在硬平板上；
    2. 用同一台摄像头从不同角度/距离/位置拍 15~30 张（越多越稳）；
    3. --square 填真实格子边长（毫米）。
"""
import argparse
import json
import os

import cv2
import numpy as np


def parse_grid(s):
    w, h = str(s).lower().split("x")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser(description="相机内参标定（棋盘格）")
    ap.add_argument("--source", default="0", help="摄像头编号 / 图片路径")
    ap.add_argument("--grid", default="9x6", help="棋盘格内角点数，如 9x6")
    ap.add_argument("--square", type=float, default=25.0, help="每格边长（毫米）")
    ap.add_argument("--out", default="camera.json", help="输出参数文件")
    args = ap.parse_args()

    gw, gh = parse_grid(args.grid)
    cap, still = open_source(args.source)

    # 棋盘格角点的世界坐标（齐平板上，z=0），按 detect 的行优先顺序生成：
    # findChessboardCorners 返回顺序为「从左到右、从上到下」，故用 mgrid[0:gw, 0:gh]
    objp = np.zeros((gw * gh, 3), np.float32)
    objp[:, :2] = np.mgrid[0:gw, 0:gh].T.reshape(-1, 2) * args.square

    obj_points, img_points = [], []
    print("s/空格 抓取样本，c 求解保存，q 退出（建议 >=15 张）")

    def grab(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (gw, gh), None)
        return gray, found, corners

    if still is not None:
        gray, found, corners = grab(still)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners)
            print("图片模式：已采集 1 个样本")
        else:
            print("图片中未检测到棋盘格，请检查 --grid 参数是否匹配")
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray, found, corners = grab(frame)
            disp = frame.copy()
            if found:
                cv2.drawChessboardCorners(disp, (gw, gh), corners, found)
            cv2.putText(disp, f"samples={len(obj_points)} (s:grab c:solve q:quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("calibrate", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("s"), ord(" ")) and found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(corners)
                print(f"已抓取样本 {len(obj_points)}")
            elif key == ord("c") and len(obj_points) >= 5:
                finish(obj_points, img_points, (gray.shape[1], gray.shape[0]), args.out)
        cap.release()

    if still is not None and len(obj_points) >= 1:
        finish(obj_points, img_points, (still.shape[1], still.shape[0]), args.out)
    cv2.destroyAllWindows()


def finish(obj_points, img_points, size, out):
    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    data = {"rms": float(ret), "size": list(size), "mtx": mtx.tolist(), "dist": dist.tolist()}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"标定完成 rms={ret:.4f}，已保存 {os.path.abspath(out)}")


def open_source(source):
    ext = os.path.splitext(str(source))[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"无法读取图片 {source}")
        return None, img
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise ValueError(f"无法打开摄像头 {source}")
    return cap, None


if __name__ == "__main__":
    main()