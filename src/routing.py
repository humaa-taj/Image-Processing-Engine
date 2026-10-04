"""Hard routing (Task 2): send each image to exactly one branch.

    route 0 (clean)     -> identity: the input is returned unchanged (no expert runs)
    route 1 (salt)      -> salt specialist
    route 2 (blur)      -> blur specialist
    route 3 (occlusion) -> occlusion specialist

The route comes either from the classifier (predicted routing = argmax of its probabilities) or
from the known test-manifest label (oracle routing).
"""
import torch

from src import config as C

BRANCH_NAMES = ["identity", "salt expert", "blur expert", "occlusion expert"]  # index = class label


@torch.no_grad()
def hard_route(x: torch.Tensor, route: torch.Tensor, experts: dict) -> torch.Tensor:
    """x: [N,3,H,W] inputs; route: [N] branch index per image; experts: {condition index: model}."""
    out = x.clone()                       # identity for every image by default (clean branch)
    for cond, model in experts.items():   # overwrite the images routed to each specialist
        mask = route == cond
        if mask.any():
            out[mask] = model(x[mask]).to(out.dtype)
    return out


def predicted_route(logits: torch.Tensor) -> torch.Tensor:
    """r = argmax_k p_k (softmax doesn't change the argmax, so logits are enough)."""
    return logits.argmax(dim=1)


assert len(BRANCH_NAMES) == len(C.CONDITIONS)
