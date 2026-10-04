"""Corruption classifier (Task 2). Its trained weights also initialize the Task 3 gate.

    input 3x128x128
      4 stages: two 3x3 convs (BN, ReLU) + 2x2 max-pool   128 -> 64 -> 32 -> 16 -> 8,  channels b, 2b, 4b, 8b
      global average pool AND global max pool, concatenated (2 x 8b numbers)
      dropout -> linear -> 4 logits  [clean, salt, blur, occlusion]

Why both poolings: average pooling summarizes the whole image (good for blur = overall lack of
fine detail), max pooling keeps the strongest local response (good for a few salt pixels or the
sharp edge of one black rectangle that averaging would dilute).
"""
import torch
import torch.nn as nn

from src.models.autoencoder import conv_bn_relu


class CorruptionClassifier(nn.Module):
    def __init__(self, base_channels: int = 32, dropout: float = 0.2, n_classes: int = 4):
        super().__init__()
        b = base_channels
        self.config = {"base_channels": base_channels, "dropout": dropout}
        layers, cin = [], 3
        for w in [b, 2 * b, 4 * b, 8 * b]:
            layers += [conv_bn_relu(cin, w), conv_bn_relu(w, w), nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (unnormalized scores). Probabilities = softmax(logits)."""
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)
