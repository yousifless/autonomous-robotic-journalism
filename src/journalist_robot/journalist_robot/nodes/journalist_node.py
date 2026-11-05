"""Journalism pipeline (Sec. III-B and III-C).

Two worker threads keep the ROS callbacks light:

selection thread   segment frames -> CLIP semantic selection -> batches of 8
description thread batches -> captured-view check (feedback to the controller)
                   -> speech transcription -> LLaVA-Interleave description
                   with the previous batch as context
                   segment END -> DeepSeek-R1 article -> acceptance score (Eq. 4)
                   -> rewrite while S < threshold (bounded) -> publish

Descriptions are produced while the robot is still recording, so only article
synthesis and checking remain once data collection ends.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from journalist_robot.core import prompts
from journalist_robot.core.frame_utils import compressed_to_bgr
from journalist_robot.core.run_recorder import RunRecorder
from journalist_robot.evaluation.acceptance import AcceptanceWeights
from journalist_robot.journalism.evidence import DescriptionBatch, SegmentEvidence
from journalist_robot.journalism.frame_selector import SemanticFrameSelector
from journalist_robot.journalism.view_check import parse_view_check
from journalist_robot.models.frame_describer import time_range
from journalist_robot.nodes.common import EVENT_QOS, SEGMENT_QOS, run_node, stamp_to_sec, use_offline_models
from journalist_robot_interfaces.msg import Article, CaptureFeedback, SegmentEvent

END_GRACE_S = 1.0          # frames of a segment may arrive shortly after its END event
MAX_EARLY_FRAMES = 2000    # frames that arrived before their START event (kept until it arrives)


@dataclass
class Frame:
    image: np.ndarray
    t: float            # seconds since the segment's first frame
    index: int          # sequential index of the selected frame within the segment


@dataclass
class SegmentState:
    evidence: SegmentEvidence
    selector: SemanticFrameSelector
    folder: str
    t0: Optional[float] = None
    batch: List[Frame] = field(default_factory=list)
    batches_queued: int = 0
    batches_dropped: int = 0
    selected: int = 0
    audio_done_s: float = 0.0   # segment-relative time up to which audio was transcribed


class AudioBuffer:
    """Mono float32 audio stamped with its receive time (wall clock)."""

    def __init__(self, sample_rate: int, keep_s: float = 900.0) -> None:
        self.sample_rate = sample_rate
        self.keep_s = keep_s
        self._chunks: List[Tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()

    def add(self, samples: np.ndarray, received_at: float) -> None:
        start = received_at - len(samples) / self.sample_rate
        with self._lock:
            self._chunks.append((start, samples))
            while self._chunks and self._chunks[0][0] < received_at - self.keep_s:
                self._chunks.pop(0)

    def slice(self, t0: float, t1: float) -> Tuple[np.ndarray, float]:
        with self._lock:
            parts = [(s, x) for (s, x) in self._chunks if s + len(x) / self.sample_rate > t0 and s < t1]
        if not parts:
            return np.zeros(0, dtype=np.float32), t0
        return np.concatenate([x for _, x in parts]), parts[0][0]


class JournalistNode(Node):
    def __init__(self) -> None:
        super().__init__("journalist")
        p = self._declare_parameters()
        self.p = p
        self.recorder = RunRecorder(p["output_root"], p["run_id"] or None)
        self._select_q: "queue.Queue" = queue.Queue()
        self._describe_q: "queue.Queue" = queue.Queue()
        self._writer_lock = threading.Lock()   # the UI and the pipeline share one LLM
        self._running = True

        lf = p["local_files_only"]
        use_offline_models(lf)
        self.get_logger().info("Loading CLIP ...")
        from journalist_robot.models.clip_encoder import ClipEncoder
        self.clip = ClipEncoder(p["clip_model_id"], local_files_only=lf)
        self.get_logger().info("Loading LLaVA-NeXT-Interleave ...")
        from journalist_robot.models.frame_describer import FrameDescriber
        self.describer = FrameDescriber(p["describer_model_id"], load_in_4bit=p["describer_4bit"],
                                        use_flash_attention=p["use_flash_attention"], local_files_only=lf)
        self.get_logger().info("Loading DeepSeek-R1 article writer ...")
        from journalist_robot.models.article_writer import ArticleWriter
        self.writer = ArticleWriter(p["writer_model_id"], load_in_4bit=p["writer_4bit"],
                                    max_new_tokens=p["writer_max_new_tokens"],
                                    min_words=p["article_min_words"], max_words=p["article_max_words"],
                                    local_files_only=lf)
        self.evaluator = None
        if p["evaluation_enabled"]:
            self.get_logger().info("Loading evaluation models ...")
            from journalist_robot.evaluation.evaluator import ArticleEvaluator
            self.evaluator = ArticleEvaluator(
                self.clip, self.describer, weights=AcceptanceWeights(*p["acceptance_weights"]),
                threshold=p["acceptance_threshold"], language_tool_url=p["language_tool_url"] or None,
                local_files_only=lf)

        self.audio: Optional[AudioBuffer] = None
        self.transcriber = None
        group = ReentrantCallbackGroup()
        if p["audio_enabled"]:
            self._setup_audio(group)

        self.ui = None
        self.index = None
        if p["ui_enabled"]:
            from journalist_robot.ui.chat_server import ChatServer
            from journalist_robot.ui.rag_index import EvidenceIndex
            self.index = EvidenceIndex()
            self.ui = ChatServer(self._answer, host=p["ui_host"], port=p["ui_port"])
            self.ui.start()
            self.get_logger().info(f"Article UI at http://{p['ui_host']}:{p['ui_port']}/")

        # Events and frames share one mutually exclusive group so that they are queued in arrival
        # order; the callbacks only enqueue, all model work happens in the worker threads.
        stream = MutuallyExclusiveCallbackGroup()
        self.create_subscription(SegmentEvent, "/journalist/segment_events", self._on_event, EVENT_QOS,
                                 callback_group=stream)
        self.create_subscription(CompressedImage, "/journalist/segment_frames", self._on_frame, SEGMENT_QOS,
                                 callback_group=stream)
        self.feedback_pub = self.create_publisher(CaptureFeedback, "/journalist/capture_feedback", EVENT_QOS)
        self.article_pub = self.create_publisher(Article, "/journalist/article", EVENT_QOS)

        self._threads = [threading.Thread(target=self._selection_loop, daemon=True),
                         threading.Thread(target=self._description_loop, daemon=True)]
        for t in self._threads:
            t.start()
        self.get_logger().info(f"Ready. Run folder: {self.recorder.root}")

    # ------------------------------------------------------------------ setup
    def _declare_parameters(self) -> dict:
        defaults = {
            "output_root": "~/journalist_robot_runs",
            "run_id": "",
            "local_files_only": False,
            "use_flash_attention": True,
            "clip_model_id": "openai/clip-vit-base-patch32",
            "describer_model_id": "llava-hf/llava-interleave-qwen-7b-hf",
            "describer_4bit": True,
            "writer_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "writer_4bit": True,
            "writer_max_new_tokens": 8192,
            "article_min_words": 800,
            "article_max_words": 1000,
            "selection_enabled": True,
            "similarity_threshold": 0.87,
            "min_selection_interval_s": 0.4,
            "batch_size": 8,
            "description_max_new_tokens": 1800,
            "max_pending_batches": 32,
            "writer_max_evidence_chars": 60000,
            "view_check_enabled": True,
            "view_check_every_n_batches": 1,
            "evaluation_enabled": True,
            "acceptance_weights": [0.35, 0.25, 0.20, 0.20],
            "acceptance_threshold": 0.85,
            "max_rewrites": 2,
            "max_key_frames": 16,
            "language_tool_url": "",
            "audio_enabled": False,
            "audio_topic": "/audio/audio",
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "asr_model_id": "openai/whisper-small",
            "asr_chunk_s": 10.0,
            "speech_output_enabled": False,
            "ui_enabled": True,
            "ui_host": "127.0.0.1",
            "ui_port": 8890,
            "shutdown_grace_s": 180.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        return {name: self.get_parameter(name).value for name in defaults}

    def _setup_audio(self, group) -> None:
        try:
            from audio_common_msgs.msg import AudioData
        except ImportError:
            self.get_logger().error("audio_enabled is true but audio_common_msgs is not installed; "
                                    "continuing without audio")
            return
        from journalist_robot.models.transcriber import SpeechTranscriber, pcm16_to_float

        self.get_logger().info("Loading Whisper ...")
        self.transcriber = SpeechTranscriber(self.p["asr_model_id"], local_files_only=self.p["local_files_only"])
        self.audio = AudioBuffer(int(self.p["audio_sample_rate"]))
        channels = int(self.p["audio_channels"])

        def _on_audio(msg: AudioData) -> None:
            self.audio.add(pcm16_to_float(msg.data, channels), time.time())

        self.create_subscription(AudioData, self.p["audio_topic"], _on_audio, 100, callback_group=group)

    # -------------------------------------------------------------- callbacks
    def _on_event(self, msg: SegmentEvent) -> None:
        self._select_q.put(("event", msg, time.time()))

    def _on_frame(self, msg: CompressedImage) -> None:
        try:
            segment_id = int(msg.header.frame_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return
        self._select_q.put(("frame", segment_id, stamp_to_sec(msg.header.stamp), bytes(msg.data)))

    # -------------------------------------------------------- selection stage
    def _selection_loop(self) -> None:
        state: Optional[SegmentState] = None
        early: deque = deque(maxlen=MAX_EARLY_FRAMES)   # frames whose START event has not arrived yet
        closing: Optional[Tuple[float, float]] = None    # (close at, collection end) after an END event
        while self._running:
            if state is not None and closing is not None and time.time() >= closing[0]:
                self._close_segment(state, closing[1])
                state, closing = None, None
            try:
                item = self._select_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                kind = item[0]
                if kind == "event":
                    msg, received = item[1], item[2]
                    if msg.type == SegmentEvent.START:
                        if state is not None:          # START without END: close the previous segment
                            self._close_segment(state, received)
                        state, closing = self._open_segment(msg, received), None
                        for frame in [f for f in early if f[1] == msg.segment_id]:
                            self._select_frame(state, frame[2], frame[3])
                        early.clear()
                    elif state is not None and (msg.type == SegmentEvent.SHUTDOWN
                                                or msg.segment_id == state.evidence.segment_id):
                        # Close a moment later, so frames still in flight are not lost.
                        closing = (time.time() + END_GRACE_S, received)
                elif kind == "frame":
                    if state is not None and item[1] == state.evidence.segment_id:
                        self._select_frame(state, item[2], item[3])
                    elif state is None or item[1] > state.evidence.segment_id:
                        early.append(item)
                elif kind == "flush" and state is not None:
                    self._close_segment(state, closing[1] if closing else item[1])
                    state, closing = None, None
            except Exception as exc:
                self.get_logger().error(f"Selection stage error: {exc}")
            finally:
                self._select_q.task_done()

    def _open_segment(self, msg: SegmentEvent, received: float) -> SegmentState:
        # The time suffix keeps folders unique if the capture manager restarts its segment ids.
        folder = f"{RunRecorder.segment_dir(msg.segment_id)}_{time.strftime('%H%M%S')}"
        evidence = SegmentEvidence(segment_id=msg.segment_id, brief=msg.mission_brief, start_time=received)
        selector = SemanticFrameSelector(self.p["similarity_threshold"], self.p["min_selection_interval_s"],
                                         enabled=self.p["selection_enabled"])
        self.get_logger().info(f"Segment {msg.segment_id}: started ({msg.pre_event_frames} pre-event frames)")
        return SegmentState(evidence=evidence, selector=selector, folder=folder)

    def _select_frame(self, state: SegmentState, stamp: float, jpeg: bytes) -> None:
        image = compressed_to_bgr(jpeg)
        if image is None:
            return
        if state.t0 is None:
            state.t0 = stamp
        t = stamp - state.t0
        embedding = self.clip.encode_image(image) if state.selector.enabled else None
        if not state.selector.consider(embedding, t):
            return
        frame = Frame(image=image, t=t, index=state.selected)
        state.selected += 1
        self.recorder.write_image(f"{state.folder}/frames/{frame.index:06d}.jpg", image)
        self._keep_key_frame(state.evidence, image, t)
        state.batch.append(frame)
        if len(state.batch) >= int(self.p["batch_size"]):
            self._queue_batch(state)

    @staticmethod
    def _keep_key_frame(evidence: SegmentEvidence, image: np.ndarray, t: float, limit: int = 300) -> None:
        small = cv2.resize(image, (640, int(image.shape[0] * 640 / image.shape[1]))) if image.shape[1] > 640 else image
        evidence.key_frames.append(small)
        evidence.key_frame_times.append(t)
        if len(evidence.key_frames) > limit:      # thin out uniformly to bound memory
            evidence.key_frames = evidence.key_frames[::2]
            evidence.key_frame_times = evidence.key_frame_times[::2]

    def _queue_batch(self, state: SegmentState) -> None:
        if not state.batch:
            return
        if self._describe_q.qsize() >= int(self.p["max_pending_batches"]):
            # The describer is far behind; bound memory instead of queuing indefinitely.
            # Selected frames are still saved in frames/; this batch is only not described.
            state.batches_dropped += 1
            self.get_logger().warn(
                f"Segment {state.evidence.segment_id}: description backlog full "
                f"({self._describe_q.qsize()} batches) - batch not described "
                f"({state.batches_dropped} so far; consider a lower similarity_threshold or larger "
                f"min_selection_interval_s)", throttle_duration_sec=10.0)
            state.batch = []
            return
        self._describe_q.put(("batch", state, list(state.batch), state.batches_queued))
        state.batches_queued += 1
        state.batch = []

    def _close_segment(self, state: SegmentState, collection_end: float) -> None:
        self._queue_batch(state)
        state.evidence.collection_end_time = collection_end
        self.recorder.write_json(f"{state.folder}/selection.json", {
            "frames_received": state.selector.seen, "frames_selected": state.selector.kept,
            "similarity_threshold": state.selector.similarity_threshold,
            "min_interval_s": state.selector.min_interval_s, "selection_enabled": state.selector.enabled,
            "batches_queued": state.batches_queued, "batches_not_described": state.batches_dropped,
        })
        self.get_logger().info(f"Segment {state.evidence.segment_id}: collection ended "
                               f"({state.selector.kept}/{state.selector.seen} frames selected)")
        self._describe_q.put(("end", state, collection_end))

    # ------------------------------------------------------ description stage
    def _description_loop(self) -> None:
        while self._running:
            try:
                item = self._describe_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if item[0] == "batch":
                    self._describe_batch(item[1], item[2], item[3])
                elif item[0] == "end":
                    self._finish_segment(item[1], item[2])
            except Exception as exc:
                self.get_logger().error(f"Description stage error: {exc}")
            finally:
                self._describe_q.task_done()

    def _segment_wall_time(self, state: SegmentState, t_rel: float) -> float:
        """Convert a segment-relative frame time to the wall clock used for audio."""
        if state.t0 is None:
            return state.evidence.start_time + t_rel
        return state.t0 + t_rel

    def _describe_batch(self, state: SegmentState, frames: List[Frame], batch_index: int) -> None:
        ev = state.evidence
        t_start = time.time()
        images = [f.image for f in frames]
        times = [f.t for f in frames]

        every = max(1, int(self.p["view_check_every_n_batches"]))
        if self.p["view_check_enabled"] and batch_index % every == 0:
            self._check_view(state, images[-2:], batch_index)

        speech = self._transcribe_until(state, times[-1]) if self.transcriber else ""
        previous = ""
        if ev.descriptions:
            prev = ev.descriptions[-1]
            previous = prev.text + (f"\nSpeech: {prev.speech}" if prev.speech else "")
        prompt = prompts.DESCRIPTION_PROMPT.format(
            n=len(images), time_range=time_range(times), brief=ev.brief or "Not provided.",
            previous=previous or "No previous context available.", speech=speech or "No speech transcribed.")
        text = self.describer.describe(images, prompt, max_new_tokens=int(self.p["description_max_new_tokens"]))

        batch = DescriptionBatch(batch_index=batch_index, frame_indices=[f.index for f in frames],
                                 timestamps_s=times, text=text, speech=speech)
        ev.descriptions.append(batch)
        elapsed = time.time() - t_start
        self.recorder.append_jsonl(f"{state.folder}/descriptions.jsonl", {
            "batch": batch_index, "frames": batch.frame_indices, "timestamps_s": times,
            "speech": speech, "description": text, "processing_time_s": elapsed})
        self.get_logger().info(f"Segment {ev.segment_id}: batch {batch_index} described "
                               f"({len(images)} frames, {elapsed:.1f}s)")

    def _check_view(self, state: SegmentState, images: List[np.ndarray], batch_index: int) -> None:
        prompt = prompts.VIEW_CHECK_PROMPT.format(brief=state.evidence.brief or "Not provided.")
        result = parse_view_check(self.describer.check(images, prompt))
        self.recorder.append_jsonl(f"{state.folder}/feedback.jsonl", {
            "batch": batch_index, "time": time.time(), "view_usable": result.view_usable,
            "on_topic": result.on_topic, "suggested_action": result.suggested_action,
            "guidance": result.guidance, "sent_to_controller": result.needs_feedback})
        if not result.needs_feedback:
            return
        msg = CaptureFeedback()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.segment_id = state.evidence.segment_id
        msg.view_usable = result.view_usable
        msg.on_topic = result.on_topic
        msg.suggested_action = result.suggested_action
        msg.guidance = result.prompt_text()
        self.feedback_pub.publish(msg)

    def _transcribe_until(self, state: SegmentState, t_rel_end: float) -> str:
        """Transcribe audio not yet processed, up to ``t_rel_end``; return speech for this span.

        Audio is stamped with this machine's clock and frames with the camera
        clock, so the robot and the GPU server must be time-synchronised
        (e.g. chrony) for speech to line up with the frames.
        """
        ev = state.evidence
        t0_rel = state.audio_done_s
        if t_rel_end - t0_rel < float(self.p["asr_chunk_s"]) * 0.5:
            return ev.speech_between(t0_rel, t_rel_end)
        wall0 = self._segment_wall_time(state, t0_rel)
        wall1 = self._segment_wall_time(state, t_rel_end)
        samples, start_wall = self.audio.slice(wall0, wall1)
        state.audio_done_s = t_rel_end
        if samples.size:
            chunks = self.transcriber.transcribe(samples, self.audio.sample_rate,
                                                 start_s=start_wall - (state.t0 or ev.start_time),
                                                 chunk_s=float(self.p["asr_chunk_s"]))
            ev.transcript.extend(chunks)
            if chunks:
                self.recorder.write_text(f"{state.folder}/transcript.txt", ev.transcript_text() + "\n")
        return ev.speech_between(t0_rel, t_rel_end)

    # ------------------------------------------------------- article + checks
    def _finish_segment(self, state: SegmentState, collection_end: float) -> None:
        ev = state.evidence
        if not ev.descriptions:
            self.get_logger().warn(f"Segment {ev.segment_id}: no frames were described; no article")
            return
        if self.transcriber:
            self._transcribe_until(state, ev.descriptions[-1].timestamps_s[-1] + float(self.p["asr_chunk_s"]))

        descriptions, transcript, timeline = self._writer_descriptions(ev), ev.transcript_text(), ev.timeline()
        key_frames = self._evaluation_frames(ev)
        with self._writer_lock:
            draft = self.writer.write(ev.brief, descriptions, transcript, timeline)
            if not draft.body:   # e.g. the reasoning trace used up the token budget
                self.get_logger().warn(f"Segment {ev.segment_id}: writer returned no article; retrying once")
                self.recorder.write_text(f"{state.folder}/drafts/failed_output_0.txt", draft.raw)
                draft = self.writer.write(ev.brief, descriptions, transcript, timeline)
        if not draft.body:
            self.recorder.write_text(f"{state.folder}/drafts/failed_output_1.txt", draft.raw)
            self.get_logger().error(f"Segment {ev.segment_id}: no article could be generated "
                                    f"(raise writer_max_new_tokens?); raw output in {state.folder}/drafts/")
            return
        report, rewrites = None, 0
        while True:
            self.recorder.write_text(f"{state.folder}/drafts/draft_{rewrites}.md", draft.markdown())
            if self.evaluator is None:
                break
            report = self.evaluator.evaluate(draft.headline, draft.lead, draft.body, key_frames,
                                             ev.description_texts())
            self.recorder.write_json(f"{state.folder}/drafts/draft_{rewrites}_evaluation.json", report.as_dict())
            self.get_logger().info(f"Segment {ev.segment_id}: draft {rewrites} S={report.score:.3f} "
                                   f"({'accepted' if report.accepted else 'below threshold'})")
            if report.accepted or rewrites >= int(self.p["max_rewrites"]):
                break
            with self._writer_lock:
                revised = self.writer.rewrite(draft, descriptions, transcript, report.problems_text())
            rewrites += 1
            if not revised.body:
                self.recorder.write_text(f"{state.folder}/drafts/draft_{rewrites}_failed_output.txt", revised.raw)
                self.get_logger().warn(f"Segment {ev.segment_id}: rewrite {rewrites} returned no article; "
                                       f"keeping the previous draft")
                rewrites -= 1
                break
            draft = revised

        generation_time = time.time() - collection_end
        accepted = report.accepted if report is not None else False
        scores = {
            "acceptance_score": report.score if report else 0.0,
            "factuality": report.factuality.score if report else 0.0,
            "relevance": report.relevance.score if report else 0.0,
            "consistency": report.consistency.score if report else 0.0,
            "language": report.language.score if report else 0.0,
        }
        self.recorder.write_text(f"{state.folder}/article.md", draft.markdown())
        self.recorder.write_json(f"{state.folder}/article.json", {
            "segment_id": ev.segment_id, "headline": draft.headline, "lead": draft.lead, "body": draft.body,
            "accepted": accepted, "evaluated": report is not None, "rewrite_count": rewrites,
            "generation_time_s": generation_time, **scores,
            "status": ("accepted" if accepted else
                       "not evaluated" if report is None else
                       f"below threshold after {rewrites} rewrites - requires human editing"),
        })
        if self.p["speech_output_enabled"]:
            from journalist_robot.models.speech_output import synthesize_to_file
            synthesize_to_file(draft.plain_text(), self.recorder.path(state.folder, "article.wav"))

        msg = Article()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.segment_id = ev.segment_id
        msg.headline, msg.body = draft.headline, f"{draft.lead}\n\n{draft.body}"
        msg.acceptance_score = float(scores["acceptance_score"])
        msg.factuality, msg.relevance = float(scores["factuality"]), float(scores["relevance"])
        msg.consistency, msg.language = float(scores["consistency"]), float(scores["language"])
        msg.accepted = bool(accepted)
        msg.rewrite_count = rewrites
        msg.generation_time_s = float(generation_time)
        msg.output_dir = self.recorder.path(state.folder)
        self.article_pub.publish(msg)
        self.get_logger().info(f"Segment {ev.segment_id}: article ready in {generation_time:.1f}s "
                               f"after collection ended -> {msg.output_dir}")

        if self.index is not None:
            self.index.add(ev.description_texts() + [c.line() for c in ev.transcript])
        if self.ui is not None:
            self.ui.publish_article({"segment_id": ev.segment_id, "headline": draft.headline,
                                     "lead": draft.lead, "body": draft.body,
                                     "acceptance_score": scores["acceptance_score"], "accepted": accepted})

    def _writer_descriptions(self, ev: SegmentEvidence) -> str:
        """Batch descriptions for the writer prompt, bounded to ``writer_max_evidence_chars``.

        Long segments would otherwise exceed the writer's context and GPU memory;
        batches are then thinned out evenly over the whole segment.
        """
        text = ev.descriptions_text()
        limit = int(self.p["writer_max_evidence_chars"])
        if limit <= 0 or len(text) <= limit:
            return text
        batches = ev.descriptions
        keep = max(1, int(len(batches) * limit / len(text)))
        idx = sorted(set(np.linspace(0, len(batches) - 1, num=keep).round().astype(int).tolist()))
        self.get_logger().warn(f"Segment {ev.segment_id}: evidence too long for the writer; using {len(idx)} "
                               f"of {len(batches)} batch descriptions, spread over the segment")
        return "\n\n".join(batches[i].formatted() for i in idx)[:limit]

    def _evaluation_frames(self, ev: SegmentEvidence) -> List[np.ndarray]:
        n = len(ev.key_frames)
        if n == 0:
            return []
        idx = np.linspace(0, n - 1, num=min(int(self.p["max_key_frames"]), n)).round().astype(int)
        return [ev.key_frames[i] for i in idx]

    def _answer(self, question: str) -> str:
        context = self.index.search(question) if self.index is not None else ""
        if not context:
            return "No recorded evidence is available yet."
        # The article pipeline has priority over UI questions for the shared writer model.
        if not self._writer_lock.acquire(timeout=2.0):
            return "The system is writing an article right now. Please ask again in a few minutes."
        try:
            return self.writer.answer(question, context)
        finally:
            self._writer_lock.release()

    def drain(self, timeout_s: float) -> bool:
        """Close an open segment and wait (bounded) until its article is finished."""
        self._select_q.put(("flush", time.time()))
        done = threading.Event()

        def _wait() -> None:
            self._select_q.join()
            self._describe_q.join()
            done.set()

        threading.Thread(target=_wait, daemon=True).start()
        return done.wait(timeout_s)

    def shutdown(self) -> None:
        self._running = False


def _shutdown(node: JournalistNode) -> None:
    grace = float(node.get_parameter("shutdown_grace_s").value)
    node.get_logger().info(f"Shutting down: finishing pending work (up to {grace:.0f}s) ...")
    if not node.drain(grace):
        node.get_logger().warn("Pending work did not finish before the grace period ended")
    node.shutdown()


def main(args=None) -> None:
    run_node(JournalistNode, num_threads=4, on_shutdown=_shutdown, args=args)


if __name__ == "__main__":
    main()
