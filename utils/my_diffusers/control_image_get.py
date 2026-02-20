from __future__ import annotations

import functools
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from rtmlib import Wholebody


class RtmlibPoseGet:
    def __init__(self, device: str = "cuda", backend: str = "onnxruntime", st: bool = True):
        self.wholebody = Wholebody(
            to_openpose=st,
            mode="performance",
            backend=backend,
            device=device,
        )

    def get_keypoints(self, img: np.ndarray):
        keypoints, scores = self.wholebody(img)
        return keypoints, scores




def draw_pose_keypoints_image(
    keypoints,
    scores,
    *,
    size: tuple[int, int],
    score_threshold: float = 0.3,
    radius: int = 4,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    background_bgr: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Draw keypoints only (no skeleton lines)."""

    w, h = size
    canvas = np.full((h, w, 3), background_bgr, dtype=np.uint8)

    if keypoints is None or len(keypoints) == 0:
        return Image.fromarray(canvas)

    for person_idx, kp in enumerate(keypoints):
        for idx, point in enumerate(kp):
            if scores is None or scores[person_idx][idx] > score_threshold:
                pt = tuple(point.astype(int))
                cv2.circle(canvas, pt, radius, color_bgr, -1)

    return Image.fromarray(canvas)


def draw_pose_skeleton_image(
    keypoints,
    scores,
    *,
    size: tuple[int, int],
    connections: Optional[Iterable[tuple[int, int]]] = None,
    score_threshold: float = 0.3,
    line_color_bgr: tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 3,
    point_color_bgr: tuple[int, int, int] = (255, 255, 255),
    point_radius: int = 4,
    background_bgr: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Draw skeleton lines + keypoints, OpenPose-style."""

    w, h = size
    canvas = np.full((h, w, 3), background_bgr, dtype=np.uint8)
    
    # common keypoint connections used to draw skeleton lines
    CONNECTIONS: list[tuple[int, int]] = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    if connections is None:
        connections = CONNECTIONS

    if keypoints is None or len(keypoints) == 0:
        return Image.fromarray(canvas)

    for person_idx, kp in enumerate(keypoints):
        # lines
        for start_idx, end_idx in connections:
            if start_idx < len(kp) and end_idx < len(kp):
                if scores is None or (
                    scores[person_idx][start_idx] > score_threshold
                    and scores[person_idx][end_idx] > score_threshold
                ):
                    start = tuple(kp[start_idx].astype(int))
                    end = tuple(kp[end_idx].astype(int))
                    cv2.line(canvas, start, end, line_color_bgr, line_thickness)

        # points
        for idx, point in enumerate(kp):
            if scores is None or scores[person_idx][idx] > score_threshold:
                pt = tuple(point.astype(int))
                cv2.circle(canvas, pt, point_radius, point_color_bgr, -1)

    return Image.fromarray(canvas)


# -------------------------
# Depth control image helpers (from test.py demo)
# -------------------------


@functools.lru_cache(maxsize=4)
def _zoe_detector(model_dir: str):
    from controlnet_aux import ZoeDetector
    return ZoeDetector.from_pretrained(model_dir)


@functools.lru_cache(maxsize=4)
def _midas_detector(model_dir: str):
    from controlnet_aux import MidasDetector
    return MidasDetector.from_pretrained(model_dir)


def get_depth_zoe(image: Image.Image, *, model_dir: str) -> Image.Image:
    """Return a Zoe depth control image (PIL)."""
    detector = _zoe_detector(model_dir)
    return detector(image)


def get_depth_midas(image: Image.Image, *, model_dir: str) -> Image.Image:
    """Return a MiDaS depth control image (PIL)."""
    detector = _midas_detector(model_dir)
    return detector(image)


def get_canny_control(image: Image.Image, low=120, high=220, blur_sigma=0.8):
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    edges = cv2.Canny(gray, low, high)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)
            
    edges_rgb = np.stack([edges] * 3, axis=2)
    return Image.fromarray(edges_rgb, mode="RGB")


def get_tile_control(image: Image.Image) -> Image.Image:
    """Return the tile control image (usually the image itself)."""
    return image.copy()

__all__ = [
    "RtmlibPoseGet",
    "draw_pose_keypoints_image",
    "draw_pose_skeleton_image",
    "get_depth_zoe",
    "get_depth_midas",
    "get_canny_control",
    "get_tile_control",
]

