"""Small RGB semantic baseline; no pretrained weights are shipped or inferred."""
import torch
from torch import nn
from torch.nn import functional as F

from . import CLASS_IDS


class SemanticBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(48, 48, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(72, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, len(CLASS_IDS), 1))

    def features(self, rgb):
        detail = self.enc(rgb)
        context = self.context(F.avg_pool2d(detail, 2, ceil_mode=True))
        context = F.interpolate(context, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((detail, context), dim=1)

    def forward(self, rgb):
        return self.head(self.features(rgb))


class OcclusionBaseline(SemanticBaseline):
    """Visible FDI plus soft-tissue occlusion, trained with separate targets."""
    def __init__(self):
        super().__init__()
        self.occlusion_head = nn.Sequential(nn.Conv2d(72, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def forward(self, rgb):
        features = self.features(rgb)
        return {"fdi_logits": self.head(features), "occlusion_logits": self.occlusion_head(features)}


def prepare(rgb, long_edge, device):
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32) / 255.0
    h, w = rgb.shape[:2]
    scale = min(1., long_edge / max(h, w))
    hw = (max(2, round(h * scale)), max(2, round(w * scale)))
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)


class ModelPredictor:
    def __init__(self, checkpoint, device="cpu", long_edge=512, *, allow_no_occlusion_head=False):
        from .data import sha256
        if type(long_edge) is not int or long_edge < 16:
            raise ValueError("long_edge must be at least 16")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_format = payload.get("format")
        if model_format not in {"stage2_semantic_baseline_v1", "stage2_occlusion_baseline_v2"} or payload.get("class_ids") != list(CLASS_IDS):
            raise ValueError("incompatible segmentation checkpoint/class mapping")
        self.has_occlusion = model_format == "stage2_occlusion_baseline_v2"
        if not self.has_occlusion and not allow_no_occlusion_head:
            raise ValueError("legacy checkpoint has no occlusion head; explicit exposed-only opt-in required")
        if self.has_occlusion and payload.get("occlusion_tissue_ids") != [101, 102, 103]:
            raise ValueError("occlusion target mapping mismatch")
        purpose, steps = payload.get("purpose"), payload.get("training_steps")
        if (purpose not in {"supervised_training", "fixture"} or type(steps) is not int or steps < 0
            or (purpose == "supervised_training" and steps == 0)):
            raise ValueError("checkpoint has no training provenance")
        self.model = OcclusionBaseline() if self.has_occlusion else SemanticBaseline()
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(device).eval()
        self.device, self.long_edge = device, long_edge
        self.kind = "fixture" if payload["purpose"] == "fixture" else "prediction"
        self.metadata = {"backend": model_format, "version": "2.0.0" if self.has_occlusion else "1.0.0",
                         "checkpoint_sha256": sha256(checkpoint), "class_ids": list(CLASS_IDS),
                         "training_steps": payload["training_steps"],
                         "training_dataset_sha256": payload.get("dataset_sha256"),
                         "score_calibration": "uncalibrated",
                         "occlusion_handling": "rgb_learned_soft_tissue" if self.has_occlusion else "explicit_exposed_only_legacy",
                         "occlusion_tissue_ids": [101, 102, 103] if self.has_occlusion else []}

    def __call__(self, rgb):
        with torch.inference_mode():
            outputs = self.model(prepare(rgb, self.long_edge, self.device))
            logits = outputs["fdi_logits"] if self.has_occlusion else outputs
            logits = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            probabilities = logits[0].softmax(0).cpu().numpy()
            if not self.has_occlusion:
                return probabilities
            occlusion = F.interpolate(outputs["occlusion_logits"], size=rgb.shape[:2], mode="bilinear", align_corners=False)
            return {"fdi_probabilities": probabilities, "occlusion_probability": occlusion[0, 0].sigmoid().cpu().numpy()}
