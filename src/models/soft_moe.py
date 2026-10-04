"""Soft mixture-of-experts restoration (Task 3).

    logits = G(x)                      gate = the Task 2 classifier architecture (initialized from it)
    w      = softmax(logits / T)       4 weights: [identity, salt, blur, occlusion], sum to 1
    x_hat  = w0*x + w1*A_salt(x) + w2*A_blur(x) + w3*A_occ(x)

Every branch runs on every image and the outputs are blended, so the whole thing is differentiable
and the gate can be trained by the reconstruction error. The temperature T controls sharpness:
small T -> nearly one-hot (close to hard routing), large T -> more evenly spread weights.
forward() returns (x_hat, w, logits); the ONNX export keeps all three outputs.
"""
import torch
import torch.nn as nn

from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier


class SoftMoE(nn.Module):
    def __init__(self, gate: CorruptionClassifier, experts: list[ConvAutoencoder], temperature: float = 1.0):
        super().__init__()
        assert len(experts) == 3, "experts must be [salt, blur, occlusion]"
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        # A buffer (not a parameter): saved with the model and exported to ONNX, but not trained.
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        w = torch.softmax(logits / self.temperature, dim=1)                       # [N, 4]
        branches = torch.stack([x] + [e(x) for e in self.experts], dim=1)          # [N, 4, 3, H, W]
        x_hat = (w[:, :, None, None, None] * branches.to(w.dtype)).sum(dim=1)     # weighted sum
        return x_hat, w, logits

    def freeze_experts(self, frozen: bool) -> None:
        for p in self.experts.parameters():
            p.requires_grad = not frozen

    def train(self, mode: bool = True):
        """Experts' BatchNorm layers always stay in eval mode (frozen running statistics).
        In the mixture every expert sees every input, so updating its BN statistics would drift them
        away from the corruption it specialises in. Their weights can still be fine-tuned."""
        super().train(mode)
        for e in self.experts:
            for m in e.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self
