"""Capture manager: dual-camera composition, pre-event buffer, segment lifecycle.

* Combines the front and rear camera views side by side at ``capture_fps``.
* Keeps the last ``pre_event_s`` seconds of combined frames in memory.
* Follows the recording trigger in ``/journalist/decision``:
    - START: publish a ``SegmentEvent.START``, then the buffered pre-event
      frames, then live frames, on ``/journalist/segment_frames``;
    - STOP : publish a ``SegmentEvent.END``.
* On shutdown it closes an open segment and publishes ``SegmentEvent.SHUTDOWN``.

Segment frames are JPEG ``sensor_msgs/CompressedImage`` messages whose
``header.frame_id`` is ``segment_<id>``.
"""

from __future__ import annotations

import array
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage

from journalist_robot.core.control_policy import TRIGGER_START, TRIGGER_STOP
from journalist_robot.core.frame_utils import bgr_to_jpeg, combine_front_rear, orient
from journalist_robot.core.mission import MissionBrief
from journalist_robot.nodes.common import EVENT_QOS, SEGMENT_QOS, run_node, subscribe_image
from journalist_robot_interfaces.msg import ControlDecision, SegmentEvent


class CaptureManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("capture_manager")
        defaults = {
            "front_image_topic": "/front_camera/image_raw/compressed",
            "rear_image_topic": "/rear_camera/image_raw/compressed",
            "image_transport": "compressed",
            "rotate_180": False,
            "use_rear_camera": True,
            "rear_max_age_s": 2.0,
            "capture_fps": 8.0,
            "pre_event_s": 10.0,
            "jpeg_quality": 90,
            "mission_brief_file": "",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = {name: self.get_parameter(name).value for name in defaults}
        self.p = p

        self.brief = MissionBrief(p["mission_brief_file"])
        maxlen = max(1, int(p["pre_event_s"] * p["capture_fps"]))
        self._pre_event: Deque[Tuple[float, bytes]] = deque(maxlen=maxlen)
        self._latest_rear: Optional[Tuple[float, np.ndarray]] = None
        self._last_capture = 0.0
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._segment_id = 0
        self._recording = False

        group = ReentrantCallbackGroup()
        subscribe_image(self, p["front_image_topic"], p["image_transport"], self._on_front, group)
        if p["use_rear_camera"]:
            subscribe_image(self, p["rear_image_topic"], p["image_transport"], self._on_rear, group)
        self.create_subscription(ControlDecision, "/journalist/decision", self._on_decision, EVENT_QOS,
                                 callback_group=group)
        self.frame_pub = self.create_publisher(CompressedImage, "/journalist/segment_frames", SEGMENT_QOS)
        self.event_pub = self.create_publisher(SegmentEvent, "/journalist/segment_events", EVENT_QOS)
        self.get_logger().info(
            f"Ready. Pre-event buffer {p['pre_event_s']:.0f}s at {p['capture_fps']:.0f} fps "
            f"({maxlen} frames); rear camera {'on' if p['use_rear_camera'] else 'off'}")

    # ---------------------------------------------------------------- cameras
    def _on_rear(self, frame: np.ndarray, stamp: float) -> None:
        with self._lock:
            self._latest_rear = (stamp, orient(frame, self.p["rotate_180"]))

    def _on_front(self, frame: np.ndarray, stamp: float) -> None:
        with self._lock:
            if stamp - self._last_capture < 1.0 / max(0.1, self.p["capture_fps"]):
                return
            self._last_capture = stamp
        front = orient(frame, self.p["rotate_180"])
        with self._lock:
            rear = None
            if self._latest_rear and abs(stamp - self._latest_rear[0]) <= self.p["rear_max_age_s"]:
                rear = self._latest_rear[1]
            jpeg = bgr_to_jpeg(combine_front_rear(front, rear), int(self.p["jpeg_quality"]))
            self._pre_event.append((stamp, jpeg))
            recording, segment_id = self._recording, self._segment_id
        if recording:
            with self._publish_lock:   # never interleave with the pre-event burst
                self._publish_frame(stamp, jpeg, segment_id)

    # --------------------------------------------------------------- segments
    def _on_decision(self, msg: ControlDecision) -> None:
        if msg.trigger == TRIGGER_START:
            self._start_segment()
        elif msg.trigger == TRIGGER_STOP:
            self._end_segment()

    def _start_segment(self) -> None:
        with self._publish_lock:
            with self._lock:
                if self._recording:
                    return
                self._segment_id += 1
                self._recording = True
                buffered = list(self._pre_event)
                segment_id = self._segment_id
            self._publish_event(SegmentEvent.START, segment_id, len(buffered))
            for stamp, jpeg in buffered:
                self._publish_frame(stamp, jpeg, segment_id)
        self.get_logger().info(f"Segment {segment_id} started with {len(buffered)} pre-event frames")

    def _end_segment(self) -> None:
        with self._publish_lock:
            with self._lock:
                if not self._recording:
                    return
                self._recording = False
                segment_id = self._segment_id
            self._publish_event(SegmentEvent.END, segment_id, 0)
        self.get_logger().info(f"Segment {segment_id} ended")

    def shutdown(self) -> None:
        self._end_segment()
        self._publish_event(SegmentEvent.SHUTDOWN, self._segment_id, 0)

    # ---------------------------------------------------------------- publish
    def _publish_frame(self, stamp: float, jpeg: bytes, segment_id: int) -> None:
        msg = CompressedImage()
        msg.header.stamp = Time(seconds=stamp).to_msg()
        msg.header.frame_id = f"segment_{segment_id}"
        msg.format = "jpeg"
        msg.data = array.array("B", jpeg)   # avoids rclpy's per-element check of a bytes value
        self.frame_pub.publish(msg)

    def _publish_event(self, kind: int, segment_id: int, pre_event_frames: int) -> None:
        msg = SegmentEvent()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type = kind
        msg.segment_id = segment_id
        msg.pre_event_frames = pre_event_frames
        msg.mission_brief = self.brief.text()
        self.event_pub.publish(msg)


def _shutdown(node: CaptureManagerNode) -> None:
    node.shutdown()
    time.sleep(0.5)  # let the final events go out


def main(args=None) -> None:
    run_node(CaptureManagerNode, num_threads=3, on_shutdown=_shutdown, args=args)


if __name__ == "__main__":
    main()
