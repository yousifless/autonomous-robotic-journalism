"""CLIP ViT-B/32 image and text embeddings.

Used for S_brief and S_novelty in Eq. (1), for semantic frame selection, and
for the visual-relevance signal of the acceptance score.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class ClipEncoder:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda",
                 local_files_only: bool = False) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_files_only)

    def encode_images(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings, shape (N, D)."""
        from journalist_robot.core.frame_utils import to_pil

        torch = self._torch
        pil = [to_pil(img) for img in images_bgr]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_image(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.encode_images([image_bgr])[0]

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """L2-normalised embeddings, shape (N, D). Texts are truncated to 77 tokens."""
        torch = self._torch
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]
