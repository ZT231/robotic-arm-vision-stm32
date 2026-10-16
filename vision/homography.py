"""S4-05 像素 -> 工作台毫米坐标的单应性映射。

相机俯视工作台时，图像上的像素点 (px,py) 与工作平面上的真实点 (Xmm,Ymm) 之间
服从平面单应变换（3x3 矩阵 H），用 4 个以上已知对应点即可求解。这样能把视觉
识别出的目标中心像素坐标，换算成机械臂抓取所需的真实毫米坐标。

两种标定方式：
    1. 棋盘格法（推荐，复用 S4-04 的棋盘格）：把打印的棋盘格平铺在工作台上，
       自动检测角点，按格子边长生成世界坐标，一次性求 H。
    2. 四点法：手动点击工作台矩形的 4 个角（顺序：原点→X方向→对角→Y方向），
       给定真实宽高求 H（无需打印棋盘格）。

用法（在 vision 目录下）：
    python homography.py --source 0 --mode chessboard --grid 9x6 --square 25 --out homography.json
    python homography.py --source 0 --mode points --width 300 --height 200 --out homography.json
    python homography.py --source worktable.png --mode points --width 300 --height 200

按键：
    棋盘格法：s 抓取当前帧求 H 并保存，q 退出
    四点法：  鼠标左键依次点 4 个角，d 求解保存，q 退出

模块函数（供下游调用）：
    load_homography(path)        读取 H
    pixel_to_world(H, px, py)    像素 -> 毫米
    world_to_pixel(H, x, y)      毫米 -> 像素
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def save_homography(H, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"homography": H.tolist()}, f, indent=2)
    print(f"已保存单应矩阵 -> {os.path.abspath(path)}")


def load_homography(path="homography.json"):
    with open(path, "r", encoding="utf-8") as f:
        return np.array(json.load(f)["homography"], dtype=np.float64)


def pixel_to_world(H, px, py):
    """像素坐标 -> 工作台毫米坐标。"""
    w = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), H)[0][0]
    return float(w[0]), float(w[1])


def world_to_pixel(H, x, y):
    """工作台毫米坐标 -> 像素坐标（H 的逆）。"""
    w = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), np.linalg.inv(H))[0][0]
    return float(w[0]), float(w[1])


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


def _solve_chessboard(frame, grid, square):
    gw, gh = grid
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (gw, gh), None)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    ids = np.arange(gw * gh).reshape(gh, gw)
    world = np.stack([(ids % gw) * square, (ids // gw) * square], axis=2)
    world = world.reshape(-1, 2).astype(np.float64)
    H, _ = cv2.findHomography(corners.reshape(-1, 2), world)
    return H


def calibrate_chessboard(source, grid, square, out):
    gw, gh = grid
    cap, still = _open(source)
    if still is not None:
        H = _solve_chessboard(still, grid, square)
        if H is None:
            print("图片中未检测到棋盘格，请检查 --grid/--square")
        else:
            save_homography(H, out)
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (gw, gh), None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, (gw, gh), corners, found)
        cv2.putText(disp, "s: solve & save  q: quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("homography-chessboard", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and found:
            H = _solve_chessboard(frame, grid, square)
            if H is not None:
                save_homography(H, out)
    cap.release()
    cv2.destroyAllWindows()


def _poly_area(pts):
    """鞋带公式计算多边形面积（像素²），用于共线退化检测。"""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def calibrate_points(source, width, height, out):
    cap, still = _open(source)
    if still is not None:
        frame = still.copy()
    else:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("无法从摄像头读取画面")
    frame = frame.copy()

    pts = []
    order = ["原点(0,0)", "X轴(W,0)", "对角(W,H)", "Y轴(0,H)"]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow("homography-points")
    cv2.setMouseCallback("homography-points", on_click)
    while True:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, f"{i + 1}.{order[i]}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(disp, "click 4 corners in order, d: solve  q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("homography-points", disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d") and len(pts) == 4:
            src = np.float32(pts[:4])
            dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            # 退化检查：四边形面积过小说明 4 点近似共线/重合，解出的 H 无意义
            area = _poly_area(src)
            if area < 200.0:
                print("警告: 4 点近似共线或重合（面积 %.0f px²），标定无效，请重新点击 4 个角" % area)
                continue
            H, _ = cv2.findHomography(src, dst)
            # 重投影误差：世界四角 -> 像素，与点击点对比，反映点击精度
            back = cv2.perspectiveTransform(
                dst.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H)).reshape(-1, 2)
            errs = np.linalg.norm(back - src, axis=1)
            max_err, mean_err = float(errs.max()), float(errs.mean())
            print("重投影误差: 平均 %.1f px / 最大 %.1f px" % (mean_err, max_err))
            if max_err > 5.0:
                print("警告: 标定精度低（最大误差 >5px），可能点击位置不准，建议重新标定")
            save_homography(H, out)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="像素->毫米 单应性标定")
    ap.add_argument("--source", default="0")
    ap.add_argument("--mode", choices=["chessboard", "points"], default="chessboard")
    ap.add_argument("--grid", default="9x6")
    ap.add_argument("--square", type=float, default=25.0, help="棋盘格每格边长（毫米）")
    ap.add_argument("--width", type=float, default=300.0, help="四点法工作台宽（毫米）")
    ap.add_argument("--height", type=float, default=200.0, help="四点法工作台高（毫米）")
    ap.add_argument("--out", default="homography.json")
    args = ap.parse_args()

    if args.mode == "chessboard":
        gw, gh = (int(x) for x in args.grid.lower().split("x"))
        calibrate_chessboard(args.source, (gw, gh), args.square, args.out)
    else:
        calibrate_points(args.source, args.width, args.height, args.out)


if __name__ == "__main__":
    main()