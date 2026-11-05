"""Perception & action loop (Sec. III-A).

Every ``decision_period_s`` the node takes the frames of the last
``decision_window_s`` from the front camera, asks VideoLLaMA3 for a movement
proposal with a newsworthiness rating, computes N_t (Eq. 1), applies the
control policy and publishes:

* ``geometry_msgs/Twist``                       on ``cmd_vel``
* ``journalist_robot_interfaces/ControlDecision`` on ``/journalist/decision``

The last command is re-published at ``cmd_publish_rate_hz`` so that bases with
a command timeout keep moving between decisions. A zero command is published
instead when no valid decision has been made for ``deadman_timeout_s``, when
the newest camera frame is older than ``max_frame_age_s``, when autonomy is
paused, and on shutdown.

These stops run on the GPU server. If the network link to the robot fails, no
command reaches the base at all, so the base must have its own command
timeout (most ROS 2 base drivers do).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Bool

from journalist_robot.core import prompts
from journalist_robot.core.control_policy import (TRIGGER_STOP, ControlPolicy, ExplorationState, PolicyConfig,
                                                  PolicyInput, RecordingTrigger, TriggerThresholds,
                                                  VelocityLimits)
from journalist_robot.core.frame_utils import orient
from journalist_robot.core.json_utils import parse_controller_output
from journalist_robot.core.mission import MissionBrief
from journalist_robot.core.newsworthiness import NewsworthinessScorer, NewsworthinessWeights
from journalist_robot.core.run_recorder import RunRecorder
from journalist_robot.core.visual_heuristics import (ForwardConsensus, motion_metric, reflex_metrics,
                                                     sampling_temperature)
from journalist_robot.nodes.common import EVENT_QOS, run_node, subscribe_image, use_offline_models
from journalist_robot_interfaces.msg import CaptureFeedback, ControlDecision


class PerceptionControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_controller")
        p = self._declare_parameters()

        self.limits = VelocityLimits(max_linear=p["max_linear"], max_angular=p["max_angular"])
        self.policy = ControlPolicy(PolicyConfig(limits=self.limits, turn_cooldown_s=p["turn_cooldown_s"],
                                                 allow_reverse=p["allow_reverse"]))
        self.trigger = RecordingTrigger(TriggerThresholds(
            start_newsworthiness=p["start_newsworthiness"], start_confidence=p["start_confidence"],
            stop_newsworthiness=p["stop_newsworthiness"], stop_confidence=p["stop_confidence"],
            max_segment_duration_s=p["max_segment_duration_s"]))
        self.exploration = ExplorationState()
        self.consensus = ForwardConsensus()
        self.scorer = NewsworthinessScorer(
            NewsworthinessWeights(*p["newsworthiness_weights"]),
            novelty_buffer_size=p["novelty_buffer_size"],
            brief_similarity_range=tuple(p["brief_similarity_range"]))
        self.brief = MissionBrief(p["mission_brief_file"])
        if not self.brief.text() and p["use_clip_newsworthiness"]:
            w = self.scorer.weights
            self.get_logger().warn(
                f"No mission brief loaded: S_brief is 0, so N_t <= {w.intrinsic + w.novelty:.2f}. Recording "
                f"starts only if this reaches start_newsworthiness ({p['start_newsworthiness']}).")
        self.recorder = RunRecorder(p["output_root"], p["run_id"] or None)

        # (camera stamp, receive time, frame); frames are oriented/resized only when selected.
        self._frames: Deque[Tuple[float, float, np.ndarray]] = deque(maxlen=128)
        self._frames_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._feedback: Optional[Tuple[float, str]] = None
        self._last_cmd = Twist()
        self._last_valid_decision = time.time()
        self._autonomy = p["autonomy_enabled"]
        self._decision_index = 0
        self._brief_digest = ""
        self._brief_embedding: Optional[np.ndarray] = None

        use_offline_models(p["local_files_only"])
        self.get_logger().info("Loading VideoLLaMA3 controller ...")
        from journalist_robot.models.videollama3_controller import VideoLLaMA3Controller
        self.controller = VideoLLaMA3Controller(
            model_id=p["controller_model_id"], use_flash_attention=p["use_flash_attention"],
            max_new_tokens=p["max_new_tokens"], local_files_only=p["local_files_only"])
        self.clip = None
        if p["use_clip_newsworthiness"]:
            self.get_logger().info("Loading CLIP for S_brief / S_novelty ...")
            from journalist_robot.models.clip_encoder import ClipEncoder
            self.clip = ClipEncoder(p["clip_model_id"], local_files_only=p["local_files_only"])

        sensors = ReentrantCallbackGroup()
        decisions = MutuallyExclusiveCallbackGroup()
        output = MutuallyExclusiveCallbackGroup()

        subscribe_image(self, p["front_image_topic"], p["image_transport"], self._on_frame, sensors)
        self.create_subscription(CaptureFeedback, "/journalist/capture_feedback", self._on_feedback,
                                 EVENT_QOS, callback_group=sensors)
        self.create_subscription(Bool, "/journalist/autonomy_enable", self._on_enable, EVENT_QOS,
                                 callback_group=sensors)
        self.cmd_pub = self.create_publisher(Twist, p["cmd_vel_topic"], 10)
        self.decision_pub = self.create_publisher(ControlDecision, "/journalist/decision", EVENT_QOS)

        self.create_timer(p["decision_period_s"], self._decide, callback_group=decisions)
        self.create_timer(1.0 / max(1.0, p["cmd_publish_rate_hz"]), self._publish_cmd, callback_group=output)

        self.get_logger().info(
            f"Ready. Run folder: {self.recorder.root} | limits v<={self.limits.max_linear} m/s, "
            f"|w|<={self.limits.max_angular} rad/s | autonomy={'on' if self._autonomy else 'off'}")

    # ------------------------------------------------------------------ setup
    def _declare_parameters(self) -> dict:
        defaults = {
            "front_image_topic": "/front_camera/image_raw/compressed",
            "image_transport": "compressed",
            "rotate_180": False,
            "cmd_vel_topic": "/cmd_vel",
            "max_frame_age_s": 3.0,
            "mission_brief_file": "",
            "output_root": "~/journalist_robot_runs",
            "run_id": "",
            "autonomy_enabled": True,
            "decision_period_s": 2.0,
            "decision_window_s": 2.0,
            "frames_per_decision": 4,
            "model_frame_width": 1280,
            "cmd_publish_rate_hz": 10.0,
            "deadman_timeout_s": 5.0,
            "max_linear": 0.4,
            "max_angular": 0.5,
            "turn_cooldown_s": 1.0,
            "allow_reverse": False,
            "start_newsworthiness": 0.7,
            "start_confidence": 0.8,
            "stop_newsworthiness": 0.3,
            "stop_confidence": 0.5,
            "max_segment_duration_s": 0.0,
            "newsworthiness_weights": [0.4, 0.4, 0.2],
            "novelty_buffer_size": 8,
            "brief_similarity_range": [0.15, 0.35],
            "use_clip_newsworthiness": True,
            "feedback_ttl_s": 10.0,
            "save_decision_frames": True,
            "controller_model_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
            "clip_model_id": "openai/clip-vit-base-patch32",
            "use_flash_attention": True,
            "max_new_tokens": 256,
            "local_files_only": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        return {name: self.get_parameter(name).value for name in defaults}

    def _param(self, name: str):
        return self.get_parameter(name).value

    # -------------------------------------------------------------- callbacks
    def _on_frame(self, frame: np.ndarray, stamp: float) -> None:
        with self._frames_lock:
            self._frames.append((stamp, time.monotonic(), frame))

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        frame = orient(frame, self._param("rotate_180"))
        width = int(self._param("model_frame_width"))
        if width > 0 and frame.shape[1] != width:
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])),
                               interpolation=cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LANCZOS4)
        return frame

    def _on_feedback(self, msg: CaptureFeedback) -> None:
        text = f"{msg.guidance} (suggested action: {msg.suggested_action})".strip()
        self._feedback = (time.time(), text)
        self.get_logger().info(f"Journalist feedback: {text}")

    def _on_enable(self, msg: Bool) -> None:
        enable = bool(msg.data)
        if enable == self._autonomy:
            return
        if enable:
            with self._cmd_lock:
                self._last_cmd = Twist()                 # no stale command after resuming
                self._last_valid_decision = time.time()
                self._autonomy = True
        else:
            with self._cmd_lock:
                self._autonomy = False
            self.stop_base(repeats=3)
            self._close_open_segment("autonomy paused")
        self.get_logger().info(f"Autonomy {'enabled' if self._autonomy else 'paused - base stopped'}")

    def _close_open_segment(self, reason: str) -> None:
        """End an open recording segment by publishing a STOP decision."""
        if not self.trigger.force_stop():
            return
        msg = ControlDecision()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.trigger = TRIGGER_STOP
        msg.movement_label = "HOLD"
        msg.reasoning = f"Recording stopped: {reason}"
        self.decision_pub.publish(msg)

    # ---------------------------------------------------------------- control
    def _window(self) -> Tuple[List[np.ndarray], float, float]:
        """Evenly spaced frames of the last decision window, their fps, and the newest frame's age (s)."""
        window_s = float(self._param("decision_window_s"))
        n = max(1, int(self._param("frames_per_decision")))
        with self._frames_lock:
            if not self._frames:
                return [], 1.0, float("inf")
            latest, received, _ = self._frames[-1]
            recent = [(t, f) for (t, _, f) in self._frames if latest - t <= window_s]
        age = time.monotonic() - received
        idx = np.linspace(0, len(recent) - 1, num=min(n, len(recent))).round().astype(int)
        chosen = [recent[i] for i in idx]
        frames = [self._prepare(f) for _, f in chosen]
        span = chosen[-1][0] - chosen[0][0]
        fps = (len(frames) - 1) / span if span > 0 and len(frames) > 1 else 1.0
        return frames, fps, age

    def _brief_text_embedding(self, brief: MissionBrief) -> Optional[np.ndarray]:
        if self.clip is None or not brief.text():
            return None
        digest = brief.digest()
        if digest != self._brief_digest:
            self._brief_embedding = self.clip.encode_text(brief.summary())
            self._brief_digest = digest
            self.scorer.reset()
            self.get_logger().info(f"Mission brief loaded ({len(brief.text())} chars, id {digest})")
        return self._brief_embedding

    def _journalist_feedback(self) -> Optional[str]:
        if self._feedback and time.time() - self._feedback[0] <= float(self._param("feedback_ttl_s")):
            return self._feedback[1]
        return None

    def _decide(self) -> None:
        now = time.time()
        if not self._autonomy:
            return
        frames, fps, age = self._window()
        if not frames:
            self.get_logger().warn("No camera frames yet", throttle_duration_sec=5.0)
            return
        if age > float(self._param("max_frame_age_s")):
            # Camera or network stalled: never act on an old view.
            self.stop_base()
            self.get_logger().warn(f"Newest camera frame is {age:.1f}s old - base stopped",
                                   throttle_duration_sec=5.0)
            return

        reflex = reflex_metrics(frames)
        consensus = self.consensus.update(not reflex["center_blocked"])
        motion = motion_metric(frames)
        brief_text = self.brief.text()
        brief_emb = self._brief_text_embedding(self.brief)

        prompt = prompts.controller_prompt(
            max_linear=self.limits.max_linear, max_angular=self.limits.max_angular,
            mission_brief=brief_text, exploration_summary=self.exploration.prompt_summary(),
            recent_commands=self.exploration.recent_history(5), reflex=reflex, motion=motion,
            journalist_feedback=self._journalist_feedback())

        try:
            out = self.controller.decide(frames, prompt, fps=fps, temperature=sampling_temperature(motion, reflex))
        except Exception as exc:
            self.get_logger().error(f"Controller inference failed: {exc}")
            return
        proposal = parse_controller_output(out.text)

        image_emb = self.clip.encode_image(frames[-1]) if self.clip is not None else None
        news = self.scorer.score(proposal["newsworthiness_score"], image_emb, brief_emb)

        result = self.policy.apply(PolicyInput(
            linear_x=proposal["linear_x"], angular_z=proposal["angular_z"],
            confidence=proposal["confidence"], newsworthiness=news.score,
            reasoning=proposal["reasoning"], inference_latency_s=out.latency_s,
            center_blocked=bool(reflex["center_blocked"]), forward_consensus=consensus,
            preferred_turn=reflex["preferred_turn"]), self.exploration, now)
        trigger = self.trigger.update(news.score, proposal["confidence"], now)
        self.exploration.update(result.label, news.score, proposal["confidence"])

        cmd = Twist()
        cmd.linear.x = float(result.linear_x)
        cmd.angular.z = float(result.angular_z)
        with self._cmd_lock:
            if not self._autonomy:      # paused while the model was running: discard this decision
                return
            self._last_cmd = cmd
            self._last_valid_decision = time.time()

        reasoning = result.reasoning(proposal["reasoning"])
        msg = ControlDecision()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.linear_x, msg.angular_z = cmd.linear.x, cmd.angular.z
        msg.confidence = float(proposal["confidence"])
        msg.newsworthiness = float(news.score)
        msg.intrinsic_score, msg.brief_relevance, msg.novelty = news.intrinsic, news.brief, news.novelty
        msg.trigger = trigger
        msg.movement_label = result.label
        msg.reasoning = reasoning
        msg.inference_latency_s = float(out.latency_s)
        self.decision_pub.publish(msg)

        self._record(frames[-1], prompt, out.text, proposal, news, result, trigger, reasoning, out.latency_s)
        self.get_logger().info(
            f"#{self._decision_index:04d} {result.label:<10} v={cmd.linear.x:+.2f} w={cmd.angular.z:+.2f} "
            f"conf={proposal['confidence']:.2f} N={news.score:.2f} "
            f"(news={news.intrinsic:.2f} brief={news.brief:.2f} nov={news.novelty:.2f}) "
            f"trigger={trigger} latency={out.latency_s:.2f}s")
        self._decision_index += 1

    def _record(self, frame, prompt, raw, proposal, news, result, trigger, reasoning, latency) -> None:
        idx = self._decision_index
        frame_file = ""
        if self._param("save_decision_frames"):
            frame_file = f"decision_frames/{idx:06d}.jpg"
            self.recorder.write_image(frame_file, frame, quality=85)
        self.recorder.append_jsonl("decisions.jsonl", {
            "index": idx, "time": time.time(), "frame": frame_file,
            "linear_x": result.linear_x, "angular_z": result.angular_z, "label": result.label,
            "proposal": proposal, "confidence": proposal["confidence"],
            "newsworthiness": news.score, "intrinsic": news.intrinsic,
            "brief_relevance": news.brief, "novelty": news.novelty,
            "trigger": trigger, "policy_notes": result.notes, "reasoning": reasoning,
            "inference_latency_s": latency, "mission_brief_id": self._brief_digest,
            "prompt": prompt, "raw_model_output": raw,
        })

    def _publish_cmd(self) -> None:
        with self._cmd_lock:
            if not self._autonomy:
                return
            if time.time() - self._last_valid_decision > float(self._param("deadman_timeout_s")):
                if self._last_cmd.linear.x != 0.0 or self._last_cmd.angular.z != 0.0:
                    self.get_logger().warn("Deadman timeout - stopping the base")
                self._last_cmd = Twist()
            # Published under the lock so that a pause/stop can never be overtaken by an older command.
            self.cmd_pub.publish(self._last_cmd)

    def stop_base(self, repeats: int = 1) -> None:
        for i in range(repeats):
            with self._cmd_lock:
                self._last_cmd = Twist()
                self.cmd_pub.publish(Twist())
            if i + 1 < repeats:
                time.sleep(0.05)

    def shutdown(self) -> None:
        with self._cmd_lock:
            self._autonomy = False
        self.stop_base(repeats=5)   # several times, in case one message is lost
        self._close_open_segment("controller shutting down")


def main(args=None) -> None:
    run_node(PerceptionControllerNode, num_threads=4, on_shutdown=lambda node: node.shutdown(), args=args)


if __name__ == "__main__":
    main()
