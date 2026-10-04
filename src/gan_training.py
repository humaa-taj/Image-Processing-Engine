"""Task 4 cGAN training loop (pix2pix recipe, with a style condition).

Per batch:
  1) Discriminator step:  L_D = 0.5 * [ BCE(D(x, y, s), 1)  +  BCE(D(x, G(x, s), s), 0) ]
                                       "D real loss"             "D fake loss"
  2) Generator step:      L_G = BCE(D(x, G(x, s), s), 1)  +  lambda_L1 * L1(y, G(x, s))
                                 "G adversarial loss"          "G L1 loss"
All four parts are logged separately (PDF). Adam(beta1 = 0.5) as in pix2pix; learning rates constant
for the first half of training, then linearly decayed to 0 (pix2pix schedule).
Precision: float32 (no AMP). The fp32 GAN fits in 4 GB and runs ~8 s/epoch, so we avoid the risk of
fp16 destabilising the adversarial game (CLAUDE.md asks to check that).
Model selection: best validation score 0.5*L1 + 0.5*(1-SSIM) between generated and real sketches
(the same fixed yardstick idea as Tasks 1-3). The adversarial loss is not a quality measure.
"""
import copy
import math
import time

import mlflow
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config as C
from src.data.fs2k import FS2KDataset
from src.metrics import objective_score, per_image_metrics
from src.models.cgan import PatchDiscriminator, UNetGenerator, init_weights
from src.tracking import TrainingDiverged
from src.utils import load_json

D_BASE = 64  # discriminator width fixed at the pix2pix standard (not exported, size doesn't matter)


def build_gan(cfg: dict, device):
    g = UNetGenerator(cfg["base_channels"], cfg["emb_dim"], cfg["dropout"]).to(device)
    d = PatchDiscriminator(D_BASE, cfg["emb_dim"]).to(device)
    g.apply(init_weights)
    d.apply(init_weights)
    return g, d


def fs2k_loaders(batch_size: int, num_workers: int = 2):
    split = load_json(C.SPLITS / "fs2k_split.json")
    train = DataLoader(FS2KDataset(split["train"], augment=True), batch_size=batch_size, shuffle=True,
                       drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    val_ds = FS2KDataset(split["val"], augment=False)
    photos, sketches, styles = zip(*[val_ds[i] for i in range(len(val_ds))])
    val = (torch.stack(photos), torch.stack(sketches), torch.tensor(styles))
    return train, val


@torch.no_grad()
def generate(g, photos: torch.Tensor, styles: torch.Tensor, device, batch_size: int = 64) -> torch.Tensor:
    g.eval()
    out = [g(photos[i:i + batch_size].to(device), styles[i:i + batch_size].to(device)).cpu()
           for i in range(0, len(photos), batch_size)]
    return torch.cat(out)


def validate_gan(g, val, device) -> dict:
    photos, sketches, styles = val
    m = per_image_metrics(generate(g, photos, styles, device), sketches)
    row = {"val_l1": m["l1"].mean().item(), "val_ssim": m["ssim"].mean().item(), "val_psnr": m["psnr"].mean().item()}
    row["val_score"] = objective_score(row["val_l1"], row["val_ssim"])
    for k in range(C.FS2K_STYLES):
        row[f"val_ssim_style{k + 1}"] = m["ssim"][styles == k].mean().item()
    return row


def lr_factor(epoch: int, epochs: int) -> float:
    """1.0 for the first half, then linear decay towards 0 (pix2pix 'linear' policy)."""
    half = epochs // 2
    return 1.0 if epoch <= half else max(0.0, 1.0 - (epoch - half) / (epochs - half + 1))


def train_gan(cfg: dict, epochs: int, device, *, trial=None, sample_fn=None, checkpoint_fn=None,
              num_workers: int = 2) -> dict:
    """cfg: lr_g, lr_d, batch_size, base_channels, dropout, emb_dim, lambda_l1."""
    g, d = build_gan(cfg, device)
    train, val = fs2k_loaders(cfg["batch_size"], num_workers)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda e: lr_factor(e + 1, epochs))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda e: lr_factor(e + 1, epochs))
    bce = torch.nn.BCEWithLogitsLoss()
    best, history = {"val_score": math.inf}, []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        g.train()
        d.train()
        sums = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        for photo, sketch, style in train:
            photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
            fake = g(photo, style)

            # 1) discriminator: real pairs -> 1, generated pairs -> 0 (fake detached: don't update G here)
            pred_real = d(photo, sketch, style)
            pred_fake = d(photo, fake.detach(), style)
            d_real = bce(pred_real, torch.ones_like(pred_real))
            d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            opt_d.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            opt_d.step()

            # 2) generator: fool D (target 1) and stay close to the real sketch (L1)
            pred_fake = d(photo, fake, style)
            g_adv = bce(pred_fake, torch.ones_like(pred_fake))
            g_l1 = F.l1_loss(fake, sketch)
            loss_g = g_adv + cfg["lambda_l1"] * g_l1
            if not torch.isfinite(loss_g) or not torch.isfinite(d_real + d_fake):
                raise TrainingDiverged(f"non-finite GAN loss at epoch {epoch}")
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            for k, v in (("d_real", d_real), ("d_fake", d_fake), ("g_adv", g_adv), ("g_l1", g_l1)):
                sums[k] += v.item()
        sched_g.step()
        sched_d.step()

        row = {"epoch": epoch, **{k: v / len(train) for k, v in sums.items()}, "lr_g": sched_g.get_last_lr()[0],
               **validate_gan(g, val, device), "epoch_s": time.perf_counter() - t0}
        history.append(row)
        mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        print(f"epoch {epoch:3d}  D real {row['d_real']:.3f} fake {row['d_fake']:.3f} | G adv {row['g_adv']:.3f} "
              f"L1 {row['g_l1']:.4f} | val_score {row['val_score']:.4f} ssim {row['val_ssim']:.4f} "
              f"({row['epoch_s']:.1f}s)", flush=True)
        if row["val_l1"] > 0.45:   # outputs nowhere near the sketches (e.g. all white/black): diverged
            raise TrainingDiverged(f"val L1 {row['val_l1']:.3f} at epoch {epoch}")
        if row["val_score"] < best["val_score"]:
            best = {**row, "state": copy.deepcopy(g.state_dict())}
        if sample_fn is not None:
            sample_fn(g, epoch)
        if checkpoint_fn is not None:
            checkpoint_fn(g, d, opt_g, opt_d, epoch, best)
        if trial is not None:
            trial.report(row["val_score"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")
    return {"history": history, "best": best, "generator": g}

