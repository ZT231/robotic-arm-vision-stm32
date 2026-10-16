"""OpenCV 颜色 + 形状识别核心（S4-02 / S4-03）。

识别流水线：
    RGB 帧 -> 裁剪 ROI -> BGR 转 HSV -> 逐颜色多段阈值分割 -> 形态学开/闭去噪
    -> findContours -> 按面积过滤 -> 形状分类 -> minAreaRect 取中心点/方向角
    -> 组装 Target 列表

设计要点（面试可讲）：
    - 用 HSV 而非 RGB 分割：颜色主要由色相 H 决定，对亮度变化、阴影更鲁棒；
    - 红色在 HSV 色相环上跨 0/180 两端，因此拆成 [0,10] 与 [170,180] 两段再并成一个 mask；
    - 开运算(先腐蚀后膨胀)去孤立噪点，闭运算(先膨胀后腐蚀)填目标内部空洞，让轮廓更干净；
    - 形状用「圆度 + 多边形逼近顶点数」联合判断，不只看顶点数（圆形逼近会有很多顶点）。
"""
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 稳定排序用的颜色顺序（与识别顺序无关，仅用于让目标列表在帧间保持稳定）
COLOR_ORDER = {"red": 0, "green": 1, "blue": 2, "yellow": 3, "pink": 4}

# 绘制叠加层时各颜色类别对应的 BGR 颜色
COLOR_BGR = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 220, 220),
    "pink": (255, 180, 220),
}


@dataclass
class Target:
    """一个被识别到的目标物。坐标均为全图像素坐标（配置 ROI 时会自动加回偏移）。"""
    color: str          # 颜色类别名，如 red/green/blue/yellow
    shape: str          # 形状：circle/triangle/square/diamond/rectangle/pentagon/polygon/unknown
    cx: float           # 中心点 x（像素）
    cy: float           # 中心点 y（像素）
    angle: float        # minAreaRect 方向角（度）
    w: float            # 最小外接矩形宽（像素）
    h: float            # 最小外接矩形高（像素）
    area: float         # 轮廓面积（像素^2）
    confidence: float   # 填充度 [0,1]：轮廓面积 / 最小外接矩形面积
    ts: float           # 时间戳（秒）
    vertices: Optional[List[Tuple[float, float]]] = None  # 轮廓逼近顶点（贴合形状绘制用，圆为 None）

    def to_dict(self) -> dict:
        return asdict(self)


def classify_shape(contour, area: float):
    """按轮廓特征分类形状，返回 (形状名, 贴合绘制顶点)，圆返回 (shape, None)。

    判断依据：
        - 圆度 = 4*pi*面积 / 周长^2：圆≈1、正方形≈0.785、三角形≈0.6；
        - 再用多边形逼近（approxPolyDP）数顶点，区分三角 / 四边 / 五边；
        - 逼近后用 _drop_spurious 去掉圆角/噪声造成的「切角」伪顶点，避免方块抖成
          pentagon/polygon、三角形抖成四边形；
        - 四边形按「边长是否等 + 内角是否直角」细分 正方形 / 菱形 / 矩形。
    """
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return "unknown", None
    circularity = 4.0 * math.pi * area / (peri * peri)
    if circularity >= 0.88:
        return "circle", None

    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    pts = _drop_spurious(pts)
    n = len(pts)

    if n == 3:
        return "triangle", _to_list(pts)
    if n == 4:
        return _quad_type(pts), _to_list(pts)
    if n == 5:
        return "pentagon", _to_list(pts)
    return "polygon", _to_list(pts)


def _drop_spurious(pts) -> np.ndarray:
    """删除圆角/噪声造成的「切角」伪顶点：相邻短边 < 最长边 20% 就合并该顶点。

    例：圆角矩形 approxPolyDP 会给出 5 个顶点（一个圆角被拆成两个），
    其中必有一条短边，删掉它即回到 4 顶点 -> 正确判为 rectangle。
    """
    pts = [tuple(p) for p in pts]
    while len(pts) > 3:
        n = len(pts)
        edges = [math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
                 for i in range(n)]
        longest = max(edges)
        idx = edges.index(min(edges))
        if edges[idx] < 0.2 * longest:
            del pts[idx]
        else:
            break
    return np.array(pts, dtype=np.float64)


def _to_list(pts) -> List[Tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in pts]


def _quad_type(pts) -> str:
    """四边形细分：正方形 / 菱形 / 矩形 / 普通四边形。

    用「对边长度均值」的比值判定矩形：对边均值能抵消单个顶点的噪声，
    避免方块因一条边偏短而被误判成长方形。
    """
    edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    pair_a = (edges[0] + edges[2]) / 2.0
    pair_b = (edges[1] + edges[3]) / 2.0
    ratio = max(pair_a, pair_b) / (min(pair_a, pair_b) + 1e-6)
    if ratio > 1.3:
        return "rectangle"  # 两对对边均值明显不等 → 长方形

    ang = _angle_at(pts, 0)          # 对边均值近似相等：看内角是 90°（方）还是斜（菱）
    return "square" if 82.0 <= ang <= 98.0 else "diamond"


def _angle_at(pts, i) -> float:
    a = pts[(i - 1) % 4]
    b = pts[i]
    c = pts[(i + 1) % 4]
    v1 = a - b
    v2 = c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


class Detector:
    """从配置文件加载阈值，对一帧图像做颜色+形状识别。"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load(config_path)
        self.min_area = float(self.config.get("min_area", 500))
        roi = self.config.get("roi")
        self.roi: Optional[Tuple[int, int, int, int]] = (
            tuple(int(v) for v in roi) if roi and len(roi) == 4 else None
        )
        self.colors = self.config.get("colors", {})
        morph = self.config.get("morph", {})
        self.open_k = self._odd(morph.get("open_ksize", 5))
        self.close_k = self._odd(morph.get("close_ksize", 5))

    @staticmethod
    def _load(config_path: Optional[str]) -> dict:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _odd(k) -> int:
        k = int(k)
        return k if k % 2 == 1 else k + 1

    def _crop(self, frame):
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    @staticmethod
    def _color_mask(hsv, ranges) -> np.ndarray:
        """把同一颜色的多段 HSV 范围合并成一个二值 mask（红色需要两段）。"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            m = cv2.inRange(
                hsv,
                np.array(low, dtype=np.uint8),
                np.array(high, dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, m)
        return mask

    def detect(self, frame) -> List[Target]:
        roi = self._crop(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
        close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))

        ox = 0 if self.roi is None else self.roi[0]
        oy = 0 if self.roi is None else self.roi[1]
        targets: List[Target] = []
        for color, spec in self.colors.items():
            mask = self._color_mask(hsv, spec["ranges"])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_se)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_se)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                shape, vertices = classify_shape(cnt, area)
                if vertices is not None:
                    vertices = [(vx + ox, vy + oy) for vx, vy in vertices]
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                rect_area = max(w * h, 1.0)
                confidence = round(min(1.0, area / rect_area), 3)
                targets.append(Target(
                    color=color,
                    shape=shape,
                    cx=float(cx) + ox,
                    cy=float(cy) + oy,
                    angle=float(angle),
                    w=float(w),
                    h=float(h),
                    area=float(area),
                    confidence=confidence,
                    ts=0.0,
                    vertices=vertices,
                ))

        targets = Detector._dedup(targets)
        # 稳定排序：按颜色固定顺序 -> 纵坐标 -> 横坐标。避免按面积排序在帧间抖动导致抓取错位
        targets.sort(key=lambda t: (COLOR_ORDER.get(t.color, 99), t.cy, t.cx))
        return targets

    @staticmethod
    def _dedup(targets: List[Target], dist_px: float = 25.0) -> List[Target]:
        """合并中心距离过近的同色目标（同一物体被 HSV 分割成多个轮廓时的去重）。

        只合并「同颜色且中心距 < dist_px」的目标，保留面积更大者；不同颜色即使叠放也不合并。
        """
        out: List[Target] = []
        for t in targets:
            merged = False
            for i, o in enumerate(out):
                if t.color == o.color and math.hypot(t.cx - o.cx, t.cy - o.cy) < dist_px:
                    if t.area > o.area:
                        out[i] = t
                    merged = True
                    break
            if not merged:
                out.append(t)
        return out


def draw_targets(frame, targets: List[Target]) -> np.ndarray:
    """把识别结果叠加到 frame 上（就地绘制并返回）：形状贴合框、中心点、方向线、短标签。

    框按轮廓本身形状绘制（圆=圆框、三角=三角框、菱形/方/矩=多边形框），而非统一最小外接矩形。
    标签做成「颜色/形状 + 坐标」两小行、放在框顶上方居中；外框用洋红色、标签黑底白字。
    """
    for t in targets:
        # 1) 形状贴合框
        if t.shape == "circle":
            radius = int((t.w + t.h) / 4)
            cv2.circle(frame, (int(t.cx), int(t.cy)), radius, (255, 0, 255), 2)
            top_y = t.cy - radius
        elif t.vertices and len(t.vertices) >= 3:
            pts = np.round(t.vertices).astype(np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
            top_y = min(p[1] for p in t.vertices)
        else:
            box = cv2.boxPoints(((t.cx, t.cy), (t.w, t.h), t.angle))
            box = np.round(box).astype(np.int32)
            cv2.drawContours(frame, [box], 0, (255, 0, 255), 2)
            top_y = min(p[1] for p in box)

        # 2) 中心点 + 方向线（长边方向：沿边抓取时的夹爪参考线）
        cv2.circle(frame, (int(t.cx), int(t.cy)), 3, (255, 0, 255), -1)
        if t.w >= t.h:
            la = t.angle
            half = t.w / 2.0
        else:
            la = t.angle + 90.0
            half = t.h / 2.0
        rad = math.radians(la)
        lx = int(t.cx + half * math.cos(rad))
        ly = int(t.cy + half * math.sin(rad))
        cv2.line(frame, (int(t.cx), int(t.cy)), (lx, ly), (0, 0, 255), 2)

        # 3) 标签（颜色/形状 + 坐标）
        line1 = f"{t.color}/{t.shape}"
        line2 = f"({t.cx:.0f},{t.cy:.0f})"
        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        text_w = max(w1, w2)
        x0 = int(t.cx) - text_w // 2
        base_y = int(top_y) - 6
        y1 = base_y - h2 - 2
        cv2.rectangle(frame, (x0 - 2, y1 - h1 - 2), (x0 + text_w + 2, base_y + 2), (0, 0, 0), -1)
        cv2.putText(frame, line1, (x0, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (x0, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA)
    return frame