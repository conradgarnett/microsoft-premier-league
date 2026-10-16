"""Person detection with RT-DETR (Apache-2.0 code and weights via Hugging Face transformers)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

MODEL = "PekingU/rtdetr_r50vd_coco_o365"


class Detector:
    def __init__(self, model: str = MODEL, device: str | None = None) -> None:
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.proc = RTDetrImageProcessor.from_pretrained(model)
        self.model = RTDetrForObjectDetection.from_pretrained(model).to(self.device).eval()
        labels = self.model.config.id2label
        self.person = next(k for k, v in labels.items() if v == "person")
        self.ball = next((k for k, v in labels.items() if v == "sports ball"), None)

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray], size: tuple[int, int] = (512, 1024), threshold: float = 0.35) -> list[dict]:
        """Batch of RGB arrays -> per image {boxes (N,4) xyxy px, scores, labels}."""
        inp = self.proc(images=images, return_tensors="pt", size={"height": size[0], "width": size[1]}).to(self.device)
        out = self.model(**inp)
        sizes = torch.tensor([im.shape[:2] for im in images])
        res = self.proc.post_process_object_detection(out, target_sizes=sizes, threshold=threshold)
        keep = {self.person, self.ball}
        result = []
        for r in res:
            m = torch.tensor([int(lbl) in keep for lbl in r["labels"]], dtype=torch.bool)
            result.append({k: r[k][m].cpu().numpy() for k in ("boxes", "scores", "labels")})
        return result
