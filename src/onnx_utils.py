"""Registry of inference models: how to rebuild each one from its checkpoint for ONNX export/verify.

Each entry: checkpoint file, a loader that returns the PyTorch model in eval mode on CPU, and the
input/output names used in the ONNX graph (the backend uses the same names).
Later phases add the classifier, specialists, soft-MoE and the GAN generator here.
"""
import functools

import torch

from src import config as C
from src.models.autoencoder import ConvAutoencoder
from src.models.cgan import UNetGenerator
from src.models.classifier import CorruptionClassifier
from src.models.soft_moe import SoftMoE

EXPERT_ORDER = ("salt", "blur", "occlusion")  # = gate branches 1, 2, 3 (branch 0 is identity)


def load_autoencoder(path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()  # eval(): dropout off, BatchNorm uses its running statistics


class ClassifierWithProbs(torch.nn.Module):
    """Export wrapper: returns logits (needed by the Task 3 gate) AND softmax probabilities (shown in the app)."""

    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(x)
        return logits, torch.softmax(logits, dim=1)


def load_classifier(path, with_probs: bool = False) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CorruptionClassifier(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ClassifierWithProbs(model).eval() if with_probs else model


def build_moe_from_task2(temperature: float, ckpt_dir=C.CHECKPOINTS, expert_tag: str = "") -> SoftMoE:
    """Task 3 initialization: gate <- Task 2 classifier, experts <- Task 2 specialists (no random start).
    expert_tag "" = v1 specialists, "_v2" = upgrade-pass specialists."""
    gate = load_classifier(ckpt_dir / "task2_classifier.pt")
    experts = [load_autoencoder(ckpt_dir / f"task2_expert_{n}{expert_tag}.pt") for n in EXPERT_ORDER]
    return SoftMoE(gate, experts, temperature)


def load_moe(path) -> SoftMoE:
    """The fine-tuned Task 3 model (whole pipeline) from its checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    gate = CorruptionClassifier(**ckpt["gate_config"])
    experts = [ConvAutoencoder(**ckpt["expert_config"]) for _ in EXPERT_ORDER]
    model = SoftMoE(gate, experts, ckpt["temperature"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def load_generator(path) -> UNetGenerator:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UNetGenerator(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()  # eval(): dropout off, BatchNorm running stats -> deterministic sketches


def fs2k_val_inputs(n: int = 48):
    """Real FS2K validation photos (16 per style) + their style ids, for generator export/verification."""
    from src.data.fs2k import FS2KDataset
    from src.utils import load_json
    records = load_json(C.SPLITS / "fs2k_split.json")["val"]
    pick = [r for k in range(3) for r in [r for r in records if r["style"] == k][:n // 3]]
    ds = FS2KDataset(pick)
    photos, _, styles = zip(*[ds[i] for i in range(len(ds))])
    return torch.stack(photos), torch.tensor(styles, dtype=torch.int64)


REGISTRY = {
    "task1_universal_ae": {
        "checkpoint": C.CHECKPOINTS / "task1_universal_ae.pt",
        "load": load_autoencoder,
        "inputs": ["input"],      # float32 [N, 3, 128, 128] in [0, 1]
        "outputs": ["output"],    # float32 [N, 3, 128, 128] in [0, 1]
    },
    "task2_classifier": {
        "checkpoint": C.CHECKPOINTS / "task2_classifier.pt",
        "load": functools.partial(load_classifier, with_probs=True),
        "inputs": ["input"],
        "outputs": ["logits", "probs"],   # [N, 4] each, order: clean, salt, blur, occlusion
    },
}
for _name in ("salt", "blur", "occlusion"):
    REGISTRY[f"task2_expert_{_name}"] = {
        "checkpoint": C.CHECKPOINTS / f"task2_expert_{_name}.pt",
        "load": load_autoencoder,
        "inputs": ["input"],
        "outputs": ["output"],
    }
REGISTRY["task3_soft_moe"] = {
    "checkpoint": C.CHECKPOINTS / "task3_soft_moe.pt",
    "load": load_moe,
    "inputs": ["input"],
    "outputs": ["output", "weights", "logits"],  # image [N,3,128,128]; weights [N,4] (identity, salt, blur, occl.)
}
REGISTRY["task4_generator"] = {
    "checkpoint": C.CHECKPOINTS / "task4_generator.pt",
    "load": load_generator,
    "inputs": ["photo", "style"],   # photo float32 [N,3,128,128] in [0,1]; style int64 [N] in {0,1,2}
    "outputs": ["sketch"],          # float32 [N,1,128,128] in [0,1]
    "example": lambda: (torch.rand(2, 3, C.IMG_SIZE, C.IMG_SIZE), torch.tensor([0, 2])),
    "real_inputs": fs2k_val_inputs,
}
