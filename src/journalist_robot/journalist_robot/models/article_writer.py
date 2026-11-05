"""DeepSeek-R1 article synthesis, revision and retrieval-augmented QA."""

from __future__ import annotations

import re
from dataclasses import dataclass

from journalist_robot.core import prompts


@dataclass
class ArticleDraft:
    headline: str
    lead: str
    body: str
    raw: str

    def markdown(self) -> str:
        return f"# {self.headline}\n\n**{self.lead}**\n\n{self.body}\n"

    def plain_text(self) -> str:
        return f"{self.headline}\n\n{self.lead}\n\n{self.body}"


def strip_reasoning(text: str) -> str:
    """Remove the ``<think>...</think>`` trace emitted by DeepSeek-R1 models."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def reasoning_unfinished(text: str) -> bool:
    """True when a ``<think>`` trace was opened but never closed (token budget exhausted)."""
    return "<think>" in text and "</think>" not in text


def parse_article(text: str) -> ArticleDraft:
    """Split model output into headline, lead and body (tolerant to format drift).

    If the reasoning trace never finished, there is no article in the output and
    an empty draft is returned (it then fails the acceptance gate instead of
    publishing chain-of-thought as an article).
    """
    if reasoning_unfinished(text):
        return ArticleDraft(headline="", lead="", body="", raw=text)
    clean = strip_reasoning(text)
    headline = re.search(r"HEADLINE:\s*(.+)", clean)
    lead = re.search(r"LEAD:\s*(.+?)(?:\n\s*BODY:|\Z)", clean, re.DOTALL)
    body = re.search(r"BODY:\s*(.+)", clean, re.DOTALL)
    if headline and body:
        return ArticleDraft(
            headline=headline.group(1).strip().strip("*#").strip(),
            lead=lead.group(1).strip() if lead else "",
            body=body.group(1).strip(),
            raw=clean,
        )
    # Fallback: first non-empty line is the headline, the rest is the body.
    lines = [ln for ln in clean.splitlines() if ln.strip()]
    if not lines:
        return ArticleDraft(headline="", lead="", body="", raw=clean)
    return ArticleDraft(headline=lines[0].strip("*# ").strip(), lead="",
                        body="\n".join(lines[1:]).strip(), raw=clean)


class ArticleWriter:
    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        load_in_4bit: bool = True,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.95,
        min_words: int = 800,
        max_words: int = 1000,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_words = min_words
        self.max_words = max_words

        # Qwen2-based models are bf16-native; fp16 can overflow on long contexts.
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kwargs = dict(device_map="auto", torch_dtype=dtype, local_files_only=local_files_only)
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    def _complete(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The chat template already contains the BOS token.
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                         temperature=temperature, top_p=self.top_p,
                                         eos_token_id=self.tokenizer.eos_token_id, pad_token_id=pad)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def write(self, brief: str, descriptions: str, transcript: str, timeline: str) -> ArticleDraft:
        prompt = prompts.ARTICLE_PROMPT.format(
            brief=brief or "No mission brief provided.",
            descriptions=descriptions,
            transcript=transcript or "No speech was transcribed.",
            timeline=timeline or "Not available.",
            min_words=self.min_words, max_words=self.max_words,
        )
        return parse_article(self._complete(prompt, self.max_new_tokens, self.temperature))

    def rewrite(self, draft: ArticleDraft, descriptions: str, transcript: str, problems: str) -> ArticleDraft:
        prompt = prompts.REWRITE_PROMPT.format(
            descriptions=descriptions,
            transcript=transcript or "No speech was transcribed.",
            draft=draft.plain_text(),
            problems=problems or "- No specific problems were listed; tighten grounding.",
        )
        return parse_article(self._complete(prompt, self.max_new_tokens, 0.5))

    def answer(self, question: str, context: str, max_new_tokens: int = 384) -> str:
        prompt = prompts.RAG_QA_PROMPT.format(context=context, question=question)
        return strip_reasoning(self._complete(prompt, max_new_tokens, 0.3))[:2000]
