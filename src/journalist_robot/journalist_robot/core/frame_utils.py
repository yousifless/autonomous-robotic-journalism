"""Image helpers: ROS image conversion (without cv_bridge) and dual-camera composition."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def compressed_to_bgr(data: bytes) -> Optional[np.ndarray]:
    """Decode a ``sensor_msgs/CompressedImage`` payload (JPEG/PNG) to BGR."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)


def raw_to_bgr(data: bytes, width: int, height: int, encoding: str, step: int) -> Optional[np.ndarray]:
    """Convert a ``sensor_msgs/Image`` payload to BGR for the common encodings."""
    buf = np.frombuffer(bytes(data), np.uint8)
    enc = encoding.lower()
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
        return img.copy() if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(height, step)[:, : width * 4].reshape(height, width, 4)
        code = cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(img, code)
    if enc in ("mono8", "8uc1"):
        img = buf.reshape(height, step)[:, :width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("yuv422", "uyvy"):
        img = buf.reshape(height, step)[:, : width * 2].reshape(height, width, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    if enc in ("yuv422_yuy2", "yuyv"):
        img = buf.reshape(height, step)[:, : width * 2].reshape(height, width, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)
    return None


def bgr_to_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return enc.tobytes()


def orient(image: np.ndarray, rotate_180: bool) -> np.ndarray:
    """Cameras on the prototype are mounted upside down; flip them if configured."""
    return cv2.rotate(image, cv2.ROTATE_180) if rotate_180 else image


def combine_front_rear(front: np.ndarray, rear: Optional[np.ndarray], label: bool = True) -> np.ndarray:
    """Place the front view (left) and rear view (right) side by side.

    Both views are scaled to the same height. If the rear view is missing,
    the front view is returned unchanged.
    """
    if rear is None:
        return front
    h = min(front.shape[0], rear.shape[0])
    if front.shape[0] != h:
        front = cv2.resize(front, (int(front.shape[1] * h / front.shape[0]), h))
    if rear.shape[0] != h:
        rear = cv2.resize(rear, (int(rear.shape[1] * h / rear.shape[0]), h))
    combined = np.hstack([front, rear])
    if label:
        mid = front.shape[1]
        cv2.line(combined, (mid, 0), (mid, h), (0, 255, 0), 3)
        cv2.putText(combined, "FRONT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "REAR", (mid + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return combined


def to_pil(image_bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
