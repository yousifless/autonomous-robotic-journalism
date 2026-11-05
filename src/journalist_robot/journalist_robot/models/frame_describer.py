"""LLaVA-NeXT-Interleave: multi-image description, view checks and consistency checks.

One model instance serves the whole journalism pipeline:

* ``describe`` - time-ordered description of a batch of frames (with the
                 previous batch's description as context);
* ``check``    - short deterministic answers, used by the captured-view check
                 and by the consistency signal of the acceptance score.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class FrameDescriber:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-interleave-qwen-7b-hf",
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

        self._torch = torch
        kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                      local_files_only=local_files_only)
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                               bnb_4bit_compute_dtype=torch.float16)
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

    def generate(self, images_bgr: Sequence[np.ndarray], prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> str:
        from journalist_robot.core.frame_utils import to_pil

        torch = self._torch
        images = [to_pil(img) for img in images_bgr]
        conversation = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in images],
        }]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images or None, padding="longest",
                                return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            # Mild repetition control for free-text descriptions only. (An n-gram ban
            # would also apply to the prompt tokens and break the JSON-formatted checks.)
            gen_kwargs.update(temperature=temperature, top_p=top_p, repetition_penalty=1.05)
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def describe(self, images_bgr: Sequence[np.ndarray], prompt: str, max_new_tokens: int = 1800) -> str:
        return self.generate(images_bgr, prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)

    def check(self, images_bgr: Sequence[np.ndarray], prompt: str, max_new_tokens: int = 200) -> str:
        """Short, deterministic answer (used for JSON-formatted checks)."""
        return self.generate(images_bgr, prompt, max_new_tokens=max_new_tokens, do_sample=False)


def time_range(timestamps_s: List[float]) -> str:
    if not timestamps_s:
        return "time unknown"
    return f"from {timestamps_s[0]:.1f}s to {timestamps_s[-1]:.1f}s"
