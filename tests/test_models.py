"""Architecture checks for the Task 1 autoencoder."""
import torch

from src.models.autoencoder import ConvAutoencoder


def test_shapes_and_range():
    m = ConvAutoencoder(base_channels=24, bottleneck_dim=512).eval()
    x = torch.rand(2, 3, 128, 128)
    z = m.encode(x)
    assert z.shape == (2, 8, 8, 8) and z[0].numel() == 512        # latent size = bottleneck_dim
    y = m(x)
    assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1


def test_output_depends_only_on_the_latent():
    """No skip connections: the decoder sees nothing but the latent code.
    So forward(x) == decode(encode(x)), and two different inputs with the same latent give the same output."""
    m = ConvAutoencoder(base_channels=24, bottleneck_dim=512).eval()
    x1, x2 = torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(m(x1), m.decode(m.encode(x1)))
        z = m.encode(x1)
        assert torch.allclose(m.decode(z), m.decode(z.clone()))
        assert (m(x1) - m(x2)).abs().max() > 0          # different inputs -> different outputs
    compression = (3 * 128 * 128) / z.numel()
    assert compression == 96


def test_classifier_shapes():
    from src.models.classifier import CorruptionClassifier
    m = CorruptionClassifier(base_channels=16).eval()
    assert m(torch.rand(3, 3, 128, 128)).shape == (3, 4)


def test_hard_route_identity_and_experts():
    """Clean route returns the input untouched; other routes use exactly their own expert."""
    from src.routing import hard_route

    class AddK(torch.nn.Module):  # fake expert that adds a recognizable constant
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return x + self.k

    x = torch.rand(4, 3, 8, 8)
    out = hard_route(x, torch.tensor([0, 1, 2, 3]), {1: AddK(1.0), 2: AddK(2.0), 3: AddK(3.0)})
    assert torch.equal(out[0], x[0])                       # identity bypass, bit-exact
    for i in (1, 2, 3):
        assert torch.allclose(out[i], x[i] + i)            # routed to expert i only


def _tiny_moe(T=1.0):
    from src.models.classifier import CorruptionClassifier
    from src.models.soft_moe import SoftMoE
    return SoftMoE(CorruptionClassifier(16), [ConvAutoencoder(24, 512) for _ in range(3)], T)


def test_soft_moe_weights_and_mixing():
    m = _tiny_moe().eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out, w, logits = m(x)
        assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-6)            # weights sum to 1
        branches = [x] + [e(x) for e in m.experts]                             # recompute the weighted sum
        manual = sum(w[:, k, None, None, None] * branches[k] for k in range(4))
        assert torch.allclose(out, manual, atol=1e-6)
        sharp = torch.softmax(logits / 0.1, 1).max(1).values                   # smaller T -> sharper
        assert (sharp >= w.max(1).values - 1e-6).all()


def test_soft_moe_expert_batchnorm_stays_frozen():
    m = _tiny_moe().train()
    bn_e = [mod for mod in m.experts.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    bn_g = [mod for mod in m.gate.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    assert all(not b.training for b in bn_e) and all(b.training for b in bn_g)


def test_cgan_shapes_and_style_conditioning():
    from src.models.cgan import PatchDiscriminator, UNetGenerator
    g, d = UNetGenerator(16, 8, 0.0).eval(), PatchDiscriminator(16, 8).eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        y = g(x, torch.tensor([0, 1]))
        assert y.shape == (2, 1, 128, 128) and 0 <= y.min() and y.max() <= 1
        assert d(x, y, torch.tensor([0, 1])).shape == (2, 1, 14, 14)      # 14x14 grid of patch logits
        # same photo, different style -> different sketch: the embedding really enters the generator
        assert (g(x[:1], torch.tensor([0])) - g(x[:1], torch.tensor([2]))).abs().max() > 0
        # and the discriminator: same pair, different style -> different patch scores
        assert (d(x[:1], y[:1], torch.tensor([0])) - d(x[:1], y[:1], torch.tensor([2]))).abs().max() > 0
