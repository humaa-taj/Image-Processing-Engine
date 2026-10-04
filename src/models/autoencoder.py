"""Convolutional denoising autoencoder with a compressed latent bottleneck (Task 1, reused in Task 2).

    input 3x128x128
      encoder: 4 stages, each halves the size and doubles the channels
               128 -> 64 -> 32 -> 16 -> 8   channels b -> 2b -> 4b -> 8b
      bottleneck: 1x1 conv squeezes 8b channels to c = bottleneck_dim / 64  -> latent 8x8xc
      decoder: 1x1 conv back to 8b, then 4 stages of (nearest upsample x2 + 3x3 convs)
    output 3x128x128 in [0, 1] (sigmoid)

Design choices:
- NO skip connections: everything the decoder knows has passed through the 8x8xc latent
  (bottleneck_dim = 256..2048 numbers vs 49,152 input values), so the network cannot copy the input.
- Convolutional (spatial) latent instead of a dense vector: a dense layer from 8x8x512 would need
  tens of millions of weights (ONNX files > 100 MB, and Task 3 puts three experts in one file),
  and keeping an 8x8 grid preserves where things are in the image.
- Upsample + conv instead of transposed conv avoids checkerboard artifacts (Odena et al., 2016).
- Dropout is applied to the latent code during training (regularizes the representation).
"""
import torch
import torch.nn as nn


def conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvAutoencoder(nn.Module):
    LATENT_SIZE = 8  # spatial size of the latent grid (128 / 2^4)

    def __init__(self, base_channels: int = 32, bottleneck_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        assert bottleneck_dim % (self.LATENT_SIZE ** 2) == 0, "bottleneck_dim must be a multiple of 64"
        b = base_channels
        widths = [b, 2 * b, 4 * b, 8 * b]
        latent_ch = bottleneck_dim // self.LATENT_SIZE ** 2
        self.config = {"base_channels": base_channels, "bottleneck_dim": bottleneck_dim, "dropout": dropout}

        enc, cin = [], 3
        for w in widths:  # each stage: strided conv (downsample) + conv
            enc += [conv_bn_relu(cin, w, stride=2), conv_bn_relu(w, w)]
            cin = w
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv2d(widths[-1], latent_ch, 1)     # the bottleneck
        self.latent_dropout = nn.Dropout(dropout)
        self.from_latent = nn.Sequential(nn.Conv2d(latent_ch, widths[-1], 1, bias=False),
                                         nn.BatchNorm2d(widths[-1]), nn.ReLU(inplace=True))

        dec, cin = [], widths[-1]
        for w in [4 * b, 2 * b, b, b]:  # 8 -> 16 -> 32 -> 64 -> 128
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), conv_bn_relu(cin, w), conv_bn_relu(w, w)]
            cin = w
        self.decoder = nn.Sequential(*dec)
        self.head = nn.Sequential(nn.Conv2d(b, 3, 3, 1, 1), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.from_latent(self.latent_dropout(z))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
