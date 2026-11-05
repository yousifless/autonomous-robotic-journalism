"""The four model-based signals of the acceptance score (Table II).

=================  ==========================================  =====================================
Signal             Purpose                                     Models
=================  ==========================================  =====================================
Factuality         grounding of claims; claim entailment       BLIP VQA; RoBERTa-MNLI
Visual relevance   evidence-text match                         CLIP image-text similarity
Consistency        semantic / temporal / spatial agreement     LLaVA-Interleave
Language           coherence; grammar; structure               Sentence-BERT; LanguageTool; rules
=================  ==========================================  =====================================
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

from journalist_robot.core.json_utils import parse_json_object
from journalist_robot.core.newsworthiness import rescale
from journalist_robot.evaluation.acceptance import SignalResult
from journalist_robot.evaluation.text_utils import split_sentences, word_count


# ---------------------------------------------------------------------------
# Factuality: visual grounding (BLIP VQA) + source adherence (RoBERTa-MNLI)
# ---------------------------------------------------------------------------

class FactualitySignal:
    def __init__(self, clip_encoder, device: str = "cuda",
                 vqa_model_id: str = "Salesforce/blip-vqa-base",
                 nli_model_id: str = "roberta-large-mnli",
                 retrieval_model_id: str = "sentence-transformers/all-mpnet-base-v2",
                 visual_weight: float = 0.6, source_weight: float = 0.4,
                 clip_min_similarity: float = 0.2, nli_threshold: float = 0.5,
                 retrieval_top_k: int = 3, max_sentences: int = 30,
                 local_files_only: bool = False) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  BlipForQuestionAnswering, BlipProcessor)

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip = clip_encoder
        self.vqa_processor = BlipProcessor.from_pretrained(vqa_model_id, local_files_only=local_files_only)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(
            vqa_model_id, local_files_only=local_files_only).to(self.device).eval()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id, local_files_only=local_files_only)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_id, local_files_only=local_files_only).to(self.device).eval()
        self.retriever = SentenceTransformer(retrieval_model_id, device=self.device)
        self.visual_weight = visual_weight
        self.source_weight = source_weight
        self.clip_min_similarity = clip_min_similarity
        self.nli_threshold = nli_threshold
        self.top_k = retrieval_top_k
        self.max_sentences = max_sentences
        label2id = {k.lower(): v for k, v in self.nli_model.config.label2id.items()}
        self._entail_id = label2id.get("entailment", 2)

    def _visual_grounding(self, sentences: List[str], frames: Sequence[np.ndarray]):
        from journalist_robot.core.frame_utils import to_pil

        torch = self._torch
        if not frames:
            return 0.5, ["no key frames available for visual grounding"]
        frame_emb = self.clip.encode_images(frames)
        sent_emb = self.clip.encode_texts(sentences)
        sims = sent_emb @ frame_emb.T
        scores, problems = [], []
        for i, sentence in enumerate(sentences):
            if float(sims[i].max()) < self.clip_min_similarity:
                continue  # sentence does not describe anything on screen; neutral
            frame = to_pil(frames[int(sims[i].argmax())])
            question = f"Is it true that {sentence.rstrip('.')}?"
            inputs = self.vqa_processor(frame, question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.vqa_model.generate(**inputs, max_new_tokens=5)
            answer = self.vqa_processor.decode(out[0], skip_special_tokens=True).strip().lower()
            ok = answer.startswith("yes")
            scores.append(1.0 if ok else 0.0)
            if not ok:
                problems.append(f'claim not visually confirmed (VQA answered "{answer}"): "{sentence}"')
        return (float(np.mean(scores)) if scores else 1.0), problems

    @staticmethod
    def _passages(descriptions: List[str], max_chars: int = 1200) -> List[str]:
        """Split descriptions into passages that fit the NLI model's 512-token input."""
        passages = []
        for description in descriptions:
            current = ""
            for sentence in split_sentences(description, min_chars=1) or [description]:
                if current and len(current) + len(sentence) + 1 > max_chars:
                    passages.append(current)
                    current = sentence[:max_chars]
                else:
                    current = f"{current} {sentence}".strip()[:max_chars]
            if current:
                passages.append(current)
        return passages

    def _source_adherence(self, sentences: List[str], descriptions: List[str]):
        torch = self._torch
        passages = self._passages(descriptions)
        if not passages:
            return 0.5, ["no frame descriptions available for source adherence"]
        src = self.retriever.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
        sent = self.retriever.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        scores, problems = [], []
        for i, sentence in enumerate(sentences):
            # Each retrieved passage is scored on its own (no truncation of the evidence);
            # a claim counts as supported by its best-supporting passage.
            top = np.argsort(-(src @ sent[i]))[: self.top_k]
            inputs = self.nli_tokenizer([passages[j] for j in top], [sentence] * len(top), return_tensors="pt",
                                        padding=True, truncation="only_first", max_length=512).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.nli_model(**inputs).logits, dim=-1)
            p = float(probs[:, self._entail_id].max())
            scores.append(p)
            if p < self.nli_threshold:
                problems.append(f'claim not entailed by the frame descriptions (p={p:.2f}): "{sentence}"')
        return (float(np.mean(scores)) if scores else 1.0), problems

    def evaluate(self, article: str, frames: Sequence[np.ndarray], descriptions: List[str]) -> SignalResult:
        sentences = split_sentences(article)[: self.max_sentences]
        if not sentences:
            return SignalResult(0.0, problems=["article is empty"])
        visual, p1 = self._visual_grounding(sentences, frames)
        source, p2 = self._source_adherence(sentences, descriptions)
        score = self.visual_weight * visual + self.source_weight * source
        return SignalResult(score, {"visual_grounding": visual, "source_adherence": source}, p1 + p2)


# ---------------------------------------------------------------------------
# Visual relevance: CLIP image-text similarity
# ---------------------------------------------------------------------------

class RelevanceSignal:
    def __init__(self, clip_encoder, similarity_range=(0.15, 0.35), max_sentences: int = 40) -> None:
        self.clip = clip_encoder
        self.similarity_range = similarity_range
        self.max_sentences = max_sentences

    def evaluate(self, article: str, frames: Sequence[np.ndarray]) -> SignalResult:
        sentences = split_sentences(article)[: self.max_sentences]
        if not sentences or not frames:
            return SignalResult(0.0, problems=["no sentences or key frames to compare"])
        sims = self.clip.encode_texts(sentences) @ self.clip.encode_images(frames).T
        best = sims.max(axis=1)
        low, high = self.similarity_range
        score = float(np.mean([rescale(float(s), low, high) for s in best]))
        weakest = [sentences[i] for i in np.argsort(best)[:2] if best[i] < low]
        problems = [f'sentence weakly related to any key frame: "{s}"' for s in weakest]
        return SignalResult(score, {"mean_max_clip_similarity": float(best.mean())}, problems)


# ---------------------------------------------------------------------------
# Consistency: LLaVA-Interleave cross-frame check
# ---------------------------------------------------------------------------

class ConsistencySignal:
    def __init__(self, describer, max_frames: int = 8) -> None:
        self.describer = describer
        self.max_frames = max_frames

    def evaluate(self, article: str, frames: Sequence[np.ndarray]) -> SignalResult:
        from journalist_robot.core import prompts

        if not frames:
            return SignalResult(0.5, problems=["no key frames for the consistency check"])
        idx = np.linspace(0, len(frames) - 1, num=min(self.max_frames, len(frames))).round().astype(int)
        answer = self.describer.check([frames[i] for i in idx],
                                      prompts.CONSISTENCY_PROMPT.format(article=article[:6000]))
        obj = parse_json_object(answer) or {}
        parts = {}
        for key in ("semantic", "temporal", "spatial"):
            try:
                parts[key] = float(np.clip(float(obj.get(key, 0.5)), 0.0, 1.0))
            except (TypeError, ValueError):
                parts[key] = 0.5
        issues = str(obj.get("issues", "")).strip()
        problems = [issues] if issues and issues.lower() != "none" else []
        if not obj:
            problems.append("consistency checker returned no parseable answer (neutral 0.5 used)")
        return SignalResult(float(np.mean(list(parts.values()))), parts, problems)


# ---------------------------------------------------------------------------
# Language: Sentence-BERT coherence, LanguageTool grammar, structure rules
# ---------------------------------------------------------------------------

class LanguageSignal:
    def __init__(self, embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 language_tool_url: Optional[str] = None, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer

        try:
            import torch
            dev = device if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
        self.embedder = SentenceTransformer(embedder_model_id, device=dev)
        self.tool = None
        try:
            import language_tool_python

            self.tool = (language_tool_python.LanguageTool("en-US", remote_server=language_tool_url)
                         if language_tool_url else language_tool_python.LanguageTool("en-US"))
        except Exception:  # LanguageTool missing or its Java server unavailable
            self.tool = None

    def _coherence(self, sentences: List[str]) -> float:
        if len(sentences) < 3:
            return 1.0
        emb = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        sims = np.sum(emb[:-1] * emb[1:], axis=1)
        return float(np.clip(sims.mean() - sims.std(), 0.0, 1.0))

    def _grammar(self, text: str):
        if self.tool is not None:
            errors = len(self.tool.check(text))
            mode = "languagetool"
        else:
            errors = len(re.findall(r" {2,}|[!?.,]{2,}", text))
            mode = "heuristic"
        return max(0.0, 1.0 - errors / 20.0), errors, mode

    @staticmethod
    def _structure(headline: str, lead: str, body: str):
        checks = {
            "has_headline": bool(headline.strip()),
            "has_lead": bool(lead.strip()),
            "body_has_3_paragraphs": len([p for p in body.split("\n\n") if p.strip()]) >= 3,
            "body_at_least_150_words": word_count(body) >= 150,
        }
        return sum(checks.values()) / len(checks), checks

    def evaluate(self, headline: str, lead: str, body: str) -> SignalResult:
        text = f"{lead}\n\n{body}"
        sentences = split_sentences(text)
        coherence = self._coherence(sentences)
        grammar, errors, mode = self._grammar(text)
        structure, checks = self._structure(headline, lead, body)
        problems = [f"structure check failed: {k}" for k, ok in checks.items() if not ok]
        if errors:
            problems.append(f"{errors} grammar/style issues ({mode})")
        score = float(np.mean([coherence, grammar, structure]))
        details = {"coherence": coherence, "grammar": grammar, "structure": structure,
                   "grammar_errors": float(errors)}
        return SignalResult(score, details, problems)
