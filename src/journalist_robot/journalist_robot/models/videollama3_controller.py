"""VideoLLaMA3 as a mission-conditioned navigation controller.

A short window of recent frames is written to a small video file (on a RAM
disk when available) and passed to VideoLLaMA3 together with the controller
prompt. Generation stops as soon as the first balanced JSON object has been
produced, which keeps per-decision latency low.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
import contextlib
from typing import List, Optional

import cv2
import numpy as np

from journalist_robot.core.json_utils import extract_balanced_json


def generated_part(sequence, prompt_ids):
    """Return only the newly generated token ids of ``sequence`` (1-D tensor).

    VideoLLaMA3 generates from ``inputs_embeds``, in which case Hugging Face
    returns only the new tokens; other code paths return prompt + new tokens.
    The prompt is stripped only when the sequence really starts with it.
    """
    n = prompt_ids.shape[0]
    if sequence.shape[0] >= n and bool((sequence[:n] == prompt_ids.to(sequence.device)).all()):
        return sequence[n:]
    return sequence


@dataclass
class ControllerOutput:
    text: str
    latency_s: float


class VideoLLaMA3Controller:
    def __init__(
        self,
        model_id: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        device: str = "cuda",
        use_flash_attention: bool = True,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
                      local_files_only=local_files_only)
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                       local_files_only=local_files_only)

        tokenizer = self.processor.tokenizer

        class _StopOnJSON(StoppingCriteria):
            """Stop generating once a complete JSON object has been emitted."""

            def __init__(self, prompt_ids) -> None:
                super().__init__()
                self.prompt_ids = prompt_ids
                self.json_text: Optional[str] = None

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                new_ids = generated_part(input_ids[0], self.prompt_ids)
                text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
                found = extract_balanced_json(text)
                if found:
                    self.json_text = found
                    return True
                return False

        self._stop_cls = _StopOnJSON

    @staticmethod
    def _write_clip(frames: List[np.ndarray], fps: float) -> str:
        directory = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        fd, path = tempfile.mkstemp(suffix=".mp4", dir=directory)
        os.close(fd)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return path

    def decide(self, frames: List[np.ndarray], prompt: str, fps: float, temperature: float) -> ControllerOutput:
        from transformers import StoppingCriteriaList

        torch = self._torch
        t0 = time.time()
        fps = max(1.0, float(fps))  # same value for the clip file and the processor
        clip_path = self._write_clip(frames, fps)
        try:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": clip_path, "fps": fps, "max_frames": len(frames)}},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self.processor(conversation=conversation, add_system_prompt=True,
                                    add_generation_prompt=True, return_tensors="pt")
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            prompt_ids = inputs["input_ids"][0]
            stop = self._stop_cls(prompt_ids)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([stop]),
                )
            if stop.json_text:
                text = stop.json_text
            else:
                text = self.processor.tokenizer.decode(generated_part(output[0], prompt_ids).tolist(),
                                                       skip_special_tokens=True)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(clip_path)
        return ControllerOutput(text=text, latency_s=time.time() - t0)
