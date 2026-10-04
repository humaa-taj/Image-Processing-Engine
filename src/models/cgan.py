"""Task 4: style-conditioned pix2pix-style cGAN (U-Net generator + PatchGAN discriminator).

Style condition: a LEARNED categorical embedding (nn.Embedding(3, emb_dim)), used inside BOTH networks:
- Generator: the embedding vector is (a) broadcast to a 128x128 map and concatenated to the photo
  (so the first layers know the style) and (b) concatenated again at the 1x1 bottleneck (so the
  decoder, which draws the strokes, also gets it directly).
- Discriminator: the embedding map is concatenated to photo + sketch, so D judges "is this a real
  sketch of THIS photo in THIS style", not just "is this a real sketch".

Generator (pix2pix U-Net, Isola et al. 2017), base width b, for 128x128 inputs:
  encoder: 7 x [4x4 conv stride 2, BN, LeakyReLU 0.2]   128 -> 1 pixel, widths b,2b,4b,8b,8b,8b,8b
  decoder: 6 x [4x4 transposed conv stride 2, BN, ReLU] + skip connection to the mirrored encoder layer
           dropout on the 3 innermost decoder layers (pix2pix uses dropout as its noise source)
  output:  1-channel sketch in [0, 1] (sigmoid); sketches are grayscale.
Skip connections ARE wanted here: photo and sketch share the same layout (eyes, outline), so the
decoder should reuse spatial detail. (Task 1's "no skips" rule is about autoencoders, not Task 4.)

Discriminator (70x70 PatchGAN): 4x4 convs b -> 2b -> 4b (stride 2) -> 8b (stride 1) -> 1, output a
14x14 grid of real/fake logits, each judging one ~70x70 patch of the 128x128 image.
"""
import torch
import torch.nn as nn


def down(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(cin, cout, dropout=0.0):
    layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def style_map(emb: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[N, E] embedding -> [N, E, h, w] constant map (same vector at every pixel)."""
    return emb[:, :, None, None].expand(-1, -1, h, w)


class UNetGenerator(nn.Module):
    def __init__(self, base_channels: int = 32, emb_dim: int = 16, dropout: float = 0.5, n_styles: int = 3):
        super().__init__()
        b = base_channels
        self.config = {"base_channels": base_channels, "emb_dim": emb_dim, "dropout": dropout}
        self.embedding = nn.Embedding(n_styles, emb_dim)
        widths = [b, 2 * b, 4 * b, 8 * b, 8 * b, 8 * b, 8 * b]          # 64, 32, 16, 8, 4, 2, 1 pixels
        self.downs = nn.ModuleList()
        cin = 3 + emb_dim
        for i, w in enumerate(widths):
            # no BatchNorm on the first layer (pix2pix) and on the 1x1 innermost layer
            self.downs.append(down(cin, w, norm=0 < i < len(widths) - 1))
            cin = w
        self.ups = nn.ModuleList()
        cin = widths[-1] + emb_dim                                        # bottleneck + style again
        for i, w in enumerate(reversed(widths[:-1])):                     # 1 -> 2 -> ... -> 64 pixels
            self.ups.append(up(cin, w, dropout if i < 3 else 0.0))
            cin = 2 * w                                                   # concatenated skip doubles it
        self.out = nn.Sequential(nn.ConvTranspose2d(cin, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, photo: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        e = self.embedding(style)
        h = torch.cat([photo, style_map(e, photo.shape[2], photo.shape[3])], dim=1)
        skips = []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = torch.cat([h, style_map(e, h.shape[2], h.shape[3])], dim=1)
        skips = skips[:-1][::-1]                                          # mirrored encoder outputs
        for i, u in enumerate(self.ups):
            h = torch.cat([u(h), skips[i]], dim=1)
        return self.out(h)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32, emb_dim: int = 16, n_styles: int = 3):
        super().__init__()
        b = base_channels
        self.embedding = nn.Embedding(n_styles, emb_dim)
        self.net = nn.Sequential(
            down(3 + 1 + emb_dim, b, norm=False),          # photo (3) + sketch (1) + style map
            down(b, 2 * b), down(2 * b, 4 * b),
            nn.Conv2d(4 * b, 8 * b, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * b), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * b, 1, 4, 1, 1))                  # logits, one per patch

    def forward(self, photo: torch.Tensor, sketch: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        e = style_map(self.embedding(style), photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], dim=1))


def init_weights(m: nn.Module) -> None:
    """pix2pix initialization: conv weights ~ N(0, 0.02), BatchNorm gamma ~ N(1, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
