#!/usr/bin/env python3
"""
Azure Kinect 총구(레이저 발사구) 전용 검출기

검출 전략(실시간/견고성 우선):
- 프레임 차이 기반 이벤트 검출(ON 순간 포착)
- 국소 최대 밝기(LoG/라플라시안) + 원형성 + 작은 면적 필터
- 깊이 범위 필터(근거리, 예: 0.2~1.2m)
- 선택적 녹색 우위성 보조

출력: 총구 중심 2D 좌표, 신뢰도, 밝기, 깊이

모든 주석/문서: 한국어
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import cv2


@dataclass
class MuzzleDetection:
    """총구 검출 결과"""
    detected: bool
    position: Tuple[int, int]
    confidence: float
    brightness: float
    depth_mm: Optional[float]


class MuzzleDetector:
    """
    Azure Kinect 총구 전용 검출기
    - 카메라별 전용 인스턴스로 사용 권장(프레임 차이 버퍼 분리)
    """

    def __init__(self,
                 near_mm: int = 200,
                 far_mm: int = 1200,
                 min_area: int = 3,
                 max_area: int = 120,
                 frame_diff_thresh: int = 25):
        """파라미터 초기화"""
        self.near_mm = near_mm
        self.far_mm = far_mm
        self.min_area = min_area
        self.max_area = max_area
        self.frame_diff_thresh = frame_diff_thresh
        self.prev_gray: Optional[np.ndarray] = None

    def _compute_motion_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """프레임 차이 기반 모션 마스크 생성"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.ones_like(gray, dtype=np.uint8) * 255
        diff = cv2.absdiff(gray, self.prev_gray)
        _, mask = cv2.threshold(diff, self.frame_diff_thresh, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray.copy()
        # 소노이즈 제거
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def detect(self, color: np.ndarray, depth: Optional[np.ndarray]) -> MuzzleDetection:
        """
        총구 포인트 검출
        Args:
            color: BGR 프레임
            depth: 깊이 프레임(mm) 또는 None
        Returns:
            MuzzleDetection
        """
        h, w = color.shape[:2]
        motion_mask = self._compute_motion_mask(color)

        # 밝기 강조 + 모션 결합
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # 적응형 임계값 후보
        thr = max(30, int(np.percentile(gray[motion_mask > 0], 75))) if np.any(motion_mask) else 60
        bright_mask = (gray >= thr).astype(np.uint8) * 255
        combined = cv2.bitwise_and(bright_mask, motion_mask)

        # 컨투어 추출
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Tuple[int, int, float, float]] = None  # (cx, cy, conf, depth)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            circ = 4.0 * np.pi * area / (peri * peri + 1e-6)
            if circ < 0.2:
                continue
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue

            # 국소 대비/라플라시안으로 점광원 특성 강화
            x1, y1 = max(0, cx - 3), max(0, cy - 3)
            x2, y2 = min(w, cx + 4), min(h, cy + 4)
            patch = gray[y1:y2, x1:x2]
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            sharpness = float(np.var(lap))
            brightness = float(np.mean(patch))

            # 깊이 필터(선택)
            depth_mm = None
            depth_ok = True
            if depth is not None and depth.size > 0:
                # 5x5 평균 깊이
                dx1, dy1 = max(0, cx - 2), max(0, cy - 2)
                dx2, dy2 = min(w, cx + 3), min(h, cy + 3)
                dpatch = depth[dy1:dy2, dx1:dx2].astype(np.float32)
                valid = dpatch[(dpatch > 0) & (dpatch < 10000)]
                if valid.size > 0:
                    depth_mm = float(np.median(valid))
                    depth_ok = (self.near_mm <= depth_mm <= self.far_mm)
                else:
                    depth_ok = False

            if not depth_ok:
                continue

            # 종합 신뢰도: 밝기 + 선예도 + 원형성
            conf = min(1.0, (brightness / 255.0) * 0.4 + min(1.0, sharpness / 200.0) * 0.4 + min(1.0, circ) * 0.2)

            if best is None or conf > best[2]:
                best = (cx, cy, conf, depth_mm if depth_mm is not None else -1.0)

        if best is None:
            return MuzzleDetection(False, (0, 0), 0.0, 0.0, None)

        cx, cy, conf, dmm = best
        return MuzzleDetection(True, (cx, cy), conf, float(gray[cy, cx]), (None if dmm < 0 else dmm))

