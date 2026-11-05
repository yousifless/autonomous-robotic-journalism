"""Helpers shared by the nodes: start-up/shutdown, image subscriptions, QoS profiles."""

from __future__ import annotations

import os
import signal
from typing import Callable, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image

from journalist_robot.core.frame_utils import compressed_to_bgr, raw_to_bgr

# Reliable, deep queue for the segment frame stream (pre-event frames arrive in a burst).
SEGMENT_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=500,
                         reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
EVENT_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=50,
                       reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)


def use_offline_models(enabled: bool) -> None:
    """With ``local_files_only``, make every Hugging Face library use the local cache only.

    Some loaders (sentence-transformers, the Whisper pipeline's tokenizer) do not
    take a ``local_files_only`` argument; the hub's offline switch covers them.
    Must be called before the model libraries are imported.
    """
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def run_node(node_factory: Callable[[], Node], num_threads: int,
             on_shutdown: Optional[Callable[[Node], None]] = None, args=None) -> None:
    """Spin a node and run ``on_shutdown`` while ROS is still up.

    rclpy's default signal handler shuts the ROS context down as soon as
    SIGINT/SIGTERM arrives, after which nothing can be published - not even a
    zero velocity command. Here that handler is disabled and SIGINT/SIGTERM
    become ``KeyboardInterrupt``, so ``on_shutdown`` can still stop the base,
    close segments and publish the last article before the context goes down.
    """
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = None
    executor = MultiThreadedExecutor(num_threads=num_threads)
    try:
        node = node_factory()
        executor.add_node(node)
        # Short waits: without rclpy's signal handler nothing wakes a blocking wait, and
        # Python only delivers the signal once control returns from the C wait call.
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if on_shutdown is not None:
                try:
                    on_shutdown(node)
                except Exception as exc:  # always reach the cleanup below
                    node.get_logger().error(f"Shutdown handler failed: {exc}")
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
        rclpy.try_shutdown()


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def subscribe_image(node: Node, topic: str, transport: str,
                    callback: Callable[[np.ndarray, float], None], callback_group=None):
    """Subscribe to a raw or compressed image topic; ``callback(bgr, stamp_s)``.

    A zero header stamp (some drivers leave it unset) is replaced by the
    node's current time.
    """

    def _stamp(msg) -> float:
        t = stamp_to_sec(msg.header.stamp)
        return t if t > 0.0 else node.get_clock().now().nanoseconds * 1e-9

    if transport == "compressed":
        def _cb(msg: CompressedImage) -> None:
            img = compressed_to_bgr(msg.data)
            if img is not None:
                callback(img, _stamp(msg))

        return node.create_subscription(CompressedImage, topic, _cb, qos_profile_sensor_data,
                                        callback_group=callback_group)

    if transport == "raw":
        def _cb_raw(msg: Image) -> None:
            try:
                img: Optional[np.ndarray] = raw_to_bgr(msg.data, msg.width, msg.height, msg.encoding, msg.step)
            except ValueError as exc:   # truncated or inconsistent image payload
                node.get_logger().warn(f"Malformed image on {topic}: {exc}", throttle_duration_sec=10.0)
                return
            if img is None:
                node.get_logger().warn(f"Unsupported image encoding '{msg.encoding}' on {topic}",
                                       throttle_duration_sec=10.0)
                return
            callback(img, _stamp(msg))

        return node.create_subscription(Image, topic, _cb_raw, qos_profile_sensor_data,
                                        callback_group=callback_group)

    raise ValueError(f"image_transport must be 'compressed' or 'raw', got '{transport}'")
