"""Small in-memory retrieval index over the recorded evidence (no FAISS needed)."""

from __future__ import annotations

import threading
from typing import List

import numpy as np


class EvidenceIndex:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id, device=device)
        self._texts: List[str] = []
        self._emb = np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        self._lock = threading.Lock()

    def add(self, texts: List[str]) -> None:
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        with self._lock:
            self._texts.extend(texts)
            self._emb = np.vstack([self._emb, emb])

    def search(self, query: str, k: int = 8, max_chars: int = 16000) -> str:
        with self._lock:
            if not self._texts:
                return ""
            q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            order = np.argsort(-(self._emb @ q))[:k]
            context = "\n\n".join(self._texts[i] for i in order)
        return context[:max_chars]
