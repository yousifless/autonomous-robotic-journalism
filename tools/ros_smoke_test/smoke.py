"""ROS 2 smoke test of the perception controller's motion and stop behaviour.

Starts the real perception_controller node (with a stand-in for VideoLLaMA3, so
no GPU is needed), feeds it camera images and checks that the robot moves on a
confident proposal, stops when the camera stalls, stops and closes the segment
when autonomy is paused, resumes, and sends zero velocity on SIGINT and SIGTERM.
Run it through run_in_container.sh.
"""
import os
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from journalist_robot.nodes.common import EVENT_QOS
from journalist_robot_interfaces.msg import ControlDecision

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


class Harness(Node):
    def __init__(self):
        super().__init__("smoke_harness")
        self.camera_on = True
        self.cmds, self.decisions = [], []
        self.img_pub = self.create_publisher(CompressedImage, "/front_camera/image_raw/compressed",
                                             qos_profile_sensor_data)
        self.enable_pub = self.create_publisher(Bool, "/journalist/autonomy_enable", EVENT_QOS)
        self.create_subscription(Twist, "/cmd_vel", lambda m: self.cmds.append((time.time(), m)), 50)
        self.decision_times = []
        self.create_subscription(ControlDecision, "/journalist/decision", self._on_decision, EVENT_QOS)
        rng = np.random.default_rng(0)
        self.jpegs = [cv2.imencode(".jpg", rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))[1].tobytes()
                      for _ in range(4)]
        self.i = 0
        self.create_timer(0.1, self._publish_image)

    def _on_decision(self, m):
        self.decisions.append(m)
        self.decision_times.append(time.time())

    def _publish_image(self):
        if not self.camera_on:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = self.jpegs[self.i % 4]
        self.i += 1
        self.img_pub.publish(msg)

    def last_cmd(self, since):
        recent = [m for t, m in self.cmds if t >= since]
        return recent[-1] if recent else None


def run(sig):
    print(f"\n=== scenario: shutdown with {sig.name} ===", flush=True)
    rclpy.init()
    h = Harness()
    ex = SingleThreadedExecutor()
    ex.add_node(h)
    stop = threading.Event()
    spinner = threading.Thread(target=lambda: [ex.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)])
    spinner.start()
    under_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_under_test.py")
    proc = subprocess.Popen([sys.executable, under_test, "--ros-args",
                             "-p", "use_clip_newsworthiness:=false", "-p", "decision_period_s:=0.5",
                             "-p", "decision_window_s:=1.0", "-p", "output_root:=/tmp/runs",
                             "-p", "max_frame_age_s:=1.0", "-p", "deadman_timeout_s:=5.0"])
    try:
        time.sleep(6.0)
        cmd = h.last_cmd(time.time() - 1.0)
        check("robot moves on a confident proposal", cmd is not None and cmd.linear.x > 0.05,
              f"(v={cmd.linear.x if cmd else None})")
        check("recording START published", any(d.trigger == "START" for d in h.decisions))

        h.camera_on = False                       # camera / network stall
        time.sleep(3.0)
        cmd = h.last_cmd(time.time() - 0.5)
        check("stale camera stops the base", cmd is not None and cmd.linear.x == 0.0 and cmd.angular.z == 0.0,
              f"(v={cmd.linear.x if cmd else None})")
        h.camera_on = True
        time.sleep(3.0)
        cmd = h.last_cmd(time.time() - 0.5)
        check("motion resumes with fresh frames", cmd is not None and cmd.linear.x > 0.05)

        n_dec = len(h.decisions)
        t_pause = time.time()
        h.enable_pub.publish(Bool(data=False))
        time.sleep(2.0)
        stops = [t for t, d in zip(h.decision_times[n_dec:], h.decisions[n_dec:]) if d.trigger == "STOP"]
        check("pause closes the open segment (STOP)", bool(stops))
        # The node publishes the zero commands just before the STOP decision; measure from there.
        t_node = stops[0] if stops else t_pause
        zero_on_pause = [m for t, m in h.cmds if t >= t_pause and m.linear.x == 0.0 and m.angular.z == 0.0]
        check("pause publishes a zero command", len(zero_on_pause) > 0)
        # Receive order across two topics is not guaranteed, so judge the cmd_vel stream itself:
        # once the first zero command after the pause has arrived, nothing may move again.
        after = [(t, m) for t, m in h.cmds if t >= t_pause]
        first_zero = next((i for i, (_, m) in enumerate(after) if m.linear.x == 0.0 and m.angular.z == 0.0), None)
        moving = [] if first_zero is None else [
            (round(t - t_pause, 3), m.linear.x, m.angular.z) for t, m in after[first_zero:]
            if m.linear.x != 0.0 or m.angular.z != 0.0]
        all_after = [(round(t - t_pause, 3), m.linear.x) for t, m in h.cmds if t >= t_pause]
        print(f"   pause: STOP received {t_node - t_pause:+.3f}s after publishing; cmds after pause: {all_after}")
        check("no motion command after the node paused", not moving, f"(moving: {moving})")

        h.enable_pub.publish(Bool(data=True))
        time.sleep(3.0)
        cmd = h.last_cmd(time.time() - 0.5)
        check("resume after pause moves again", cmd is not None and cmd.linear.x > 0.05)

        t_sig = time.time()
        proc.send_signal(sig)
        code = proc.wait(timeout=15)
        time.sleep(0.5)
        final = [m for t, m in h.cmds if t >= t_sig]
        check(f"process exits on {sig.name}", code is not None, f"(exit code {code})")
        check(f"zero command sent on {sig.name}",
              bool(final) and final[-1].linear.x == 0.0 and final[-1].angular.z == 0.0,
              f"({len(final)} msgs after signal)")
    finally:
        if proc.poll() is None:
            proc.kill()
        stop.set()
        spinner.join()
        h.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    for s in (signal.SIGINT, signal.SIGTERM):
        run(s)
    print("\nRESULT:", "ALL PASSED" if not FAILURES else f"FAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)
