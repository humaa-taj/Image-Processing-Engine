"""Time real training epochs on this GPU to calibrate the time estimates in the plan.

The models here are THROWAWAY stand-ins of roughly the size we expect (the real architectures are
written in Phases 2 and 5). We measure:
  1. data pipeline speed (runtime corruption on CPU) for num_workers = 0 and 2
  2. one autoencoder epoch (2,944 train images, AMP) + one validation pass (736 images)
  3. one GAN epoch on FS2K (899 pairs, U-Net G + PatchGAN D, batch 8)
Results -> artifacts/results/benchmark_epoch.json and MLflow experiment 'phase1-benchmark'
Run:  .venv\\Scripts\\python scripts/benchmark_epoch.py
"""
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import fs2k, pet  # noqa: E402
from src.losses import restoration_loss  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.utils import get_device, load_json, seed_everything  # noqa: E402


# ------------------------------------------------------------------ throwaway models (benchmark only)
def conv_block(cin, cout, down=True):
    conv = nn.Conv2d(cin, cout, 4, 2, 1) if down else nn.ConvTranspose2d(cin, cout, 4, 2, 1)
    return nn.Sequential(conv, nn.BatchNorm2d(cout), nn.LeakyReLU(0.2) if down else nn.ReLU())


class BenchAE(nn.Module):
    """128 -> 8x8x256 -> dense bottleneck 256 -> back to 128. ~ the Task 1 size we expect."""
    def __init__(self, ch=32, bottleneck=256):
        super().__init__()
        self.enc = nn.Sequential(conv_block(3, ch), conv_block(ch, ch * 2), conv_block(ch * 2, ch * 4),
                                 conv_block(ch * 4, ch * 8))
        self.fc_in = nn.Linear(ch * 8 * 8 * 8, bottleneck)
        self.fc_out = nn.Linear(bottleneck, ch * 8 * 8 * 8)
        self.dec = nn.Sequential(conv_block(ch * 8, ch * 4, False), conv_block(ch * 4, ch * 2, False),
                                 conv_block(ch * 2, ch, False), nn.ConvTranspose2d(ch, 3, 4, 2, 1), nn.Sigmoid())
        self.ch = ch

    def forward(self, x):
        z = self.fc_in(self.enc(x).flatten(1))
        return self.dec(self.fc_out(z).view(-1, self.ch * 8, 8, 8))


class BenchUNet(nn.Module):
    """pix2pix-like U-Net, 128 -> 1x1, base 64 channels, style map as extra input channels."""
    def __init__(self, ch=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(3, emb)
        widths = [ch, ch * 2, ch * 4, ch * 8, ch * 8, ch * 8, ch * 8]
        self.downs = nn.ModuleList()
        cin = 3 + emb
        for w in widths:
            self.downs.append(conv_block(cin, w))
            cin = w
        self.ups = nn.ModuleList()
        for i, w in enumerate(reversed(widths[:-1])):
            self.ups.append(conv_block(cin if i == 0 else cin * 2, w, down=False))
            cin = w
        self.out = nn.ConvTranspose2d(cin * 2, 1, 4, 2, 1)

    def forward(self, x, s):
        e = self.emb(s)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        h, skips = torch.cat([x, e], 1), []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        skips = skips[:-1][::-1]
        for i, u in enumerate(self.ups):
            h = u(h if i == 0 else torch.cat([h, skips[i - 1]], 1))
        return torch.sigmoid(self.out(torch.cat([h, skips[-1]], 1)))


class BenchPatchD(nn.Module):
    def __init__(self, ch=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(3, emb)
        self.net = nn.Sequential(nn.Conv2d(3 + 1 + emb, ch, 4, 2, 1), nn.LeakyReLU(0.2),
                                 conv_block(ch, ch * 2), conv_block(ch * 2, ch * 4),
                                 nn.Conv2d(ch * 4, ch * 8, 4, 1, 1), nn.BatchNorm2d(ch * 8), nn.LeakyReLU(0.2),
                                 nn.Conv2d(ch * 8, 1, 4, 1, 1))

    def forward(self, x, y, s):
        e = self.emb(s)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, y, e], 1))


# ------------------------------------------------------------------ benchmarks
def time_loader(loader, max_batches=None):
    t = time.perf_counter()
    n = 0
    for i, batch in enumerate(loader):
        n += batch[0].shape[0]
        if max_batches and i + 1 >= max_batches:
            break
    return n / (time.perf_counter() - t)


def bench_data(train_imgs):
    out = {}
    for workers in (0, 2):
        ds = pet.PetTrainDataset(train_imgs)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=workers,
                            persistent_workers=workers > 0)
        if workers:
            time_loader(loader, 3)  # warm-up: worker start-up on Windows (spawn) is slow
        out[f"images_per_sec_workers{workers}"] = round(time_loader(loader), 1)
        print(f"  data pipeline, num_workers={workers}: {out[f'images_per_sec_workers{workers}']} img/s")
    return out


def bench_ae(train_imgs, device, workers):
    model = BenchAE().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    scaler = torch.amp.GradScaler()
    loader = DataLoader(pet.PetTrainDataset(train_imgs), batch_size=32, shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    val_loader = DataLoader(pet.load_manifest_dataset("val"), batch_size=64, num_workers=0)
    torch.cuda.reset_peak_memory_stats()
    results = {"params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    for epoch in range(2):  # epoch 0 includes cuDNN autotune / worker start-up; epoch 1 is the clean number
        t = time.perf_counter()
        model.train()
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = restoration_loss(pred, y, alpha=0.8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        train_s = time.perf_counter() - t
        t = time.perf_counter()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x, y, _, _ in val_loader:
                model(x.to(device))
        torch.cuda.synchronize()
        val_s = time.perf_counter() - t
        print(f"  AE epoch {epoch}: train {train_s:.1f}s, val {val_s:.1f}s, loss {loss.item():.4f}")
    results.update(train_epoch_s=round(train_s, 1), val_pass_s=round(val_s, 1),
                   peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return results


def bench_gan(device, workers):
    split = load_json(C.SPLITS / "fs2k_split.json")
    loader = DataLoader(fs2k.FS2KDataset(split["train"], augment=True), batch_size=8, shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=True, drop_last=True)
    G, D = BenchUNet().to(device), BenchPatchD().to(device)
    opt_g = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    results = {"G_params_M": round(sum(p.numel() for p in G.parameters()) / 1e6, 2),
               "D_params_M": round(sum(p.numel() for p in D.parameters()) / 1e6, 2)}
    for epoch in range(2):
        t = time.perf_counter()
        for photo, sketch, style in loader:  # fp32 on purpose: GAN + AMP is checked separately later
            photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
            fake = G(photo, style)
            d_real, d_fake = D(photo, sketch, style), D(photo, fake.detach(), style)
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            d_fake = D(photo, fake, style)
            loss_g = bce(d_fake, torch.ones_like(d_fake)) + 100 * F.l1_loss(fake, sketch)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
        torch.cuda.synchronize()
        epoch_s = time.perf_counter() - t
        print(f"  GAN epoch {epoch}: {epoch_s:.1f}s (D {loss_d.item():.3f}, G {loss_g.item():.3f})")
    results.update(epoch_s=round(epoch_s, 1), peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return results


def main():
    device = get_device()
    seed_everything(C.SEED)
    torch.backends.cudnn.benchmark = True
    train_imgs = pet.load_cached("trainval", pet.load_split()["train"])
    print("1) data pipeline")
    data = bench_data(train_imgs)
    workers = 2 if data["images_per_sec_workers2"] > 1.2 * data["images_per_sec_workers0"] else 0
    print(f"2) autoencoder (num_workers={workers})")
    ae = bench_ae(train_imgs, device, workers)
    print(f"3) GAN (num_workers={workers})")
    gan = bench_gan(device, workers)
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "num_workers": workers,
           "data": data, "autoencoder_bs32_amp": ae, "gan_bs8_fp32": gan}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    path = C.RESULTS / "benchmark_epoch.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    setup_mlflow("phase1-benchmark")
    with mlflow.start_run(run_name="epoch-timing"):
        mlflow.log_params({"gpu": out["gpu"], "torch": out["torch"], "num_workers": workers})
        for section in ("data", "autoencoder_bs32_amp", "gan_bs8_fp32"):
            mlflow.log_metrics({f"{section}.{k}": v for k, v in out[section].items()})
        mlflow.log_artifact(str(path))
    print("logged to MLflow experiment 'phase1-benchmark'")


if __name__ == "__main__":
    main()
