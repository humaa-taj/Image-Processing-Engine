"""Architecture diagrams for the report (drawn with matplotlib, no extra tools).

Run:  .venv\\Scripts\\python scripts/make_diagrams.py
-> docs/figures/diagram_task1_autoencoder.png, diagram_task3_soft_moe.png,
   diagram_task4_cgan.png, diagram_app.png
Sizes shown are the final (Optuna-selected) configurations.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

BLUE, GREY, GREEN, ORANGE, PURPLE, INK = "#d6e4f0", "#eceff3", "#dcefe1", "#fbe5cf", "#e8def3", "#1f2937"


def box(ax, x, y, w, h, text, color=GREY, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#4b5563", lw=1.15))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, style="-|>", color="#374151", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, ha="center", fontsize=7.5, color="#4b5563",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    out = C.FIGURES / name
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print("saved", out.relative_to(C.ROOT))


def task1():
    fig, ax = canvas(16, 5.0)
    cy = 2.7
    # (label, spatial size -> box height, colour)
    stages = [("Input\n3×128×128", 128, GREY), ("Enc 1\n64×64×64", 64, BLUE), ("Enc 2\n128×32×32", 32, BLUE),
              ("Enc 3\n256×16×16", 16, BLUE), ("Enc 4\n512×8×8", 8, BLUE), ("Latent\n32×8×8\n(2,048 values)", 8, ORANGE),
              ("Dec 1\n256×16×16", 16, GREEN), ("Dec 2\n128×32×32", 32, GREEN), ("Dec 3\n64×64×64", 64, GREEN),
              ("Dec 4\n64×128×128", 128, GREEN), ("Output\n3×128×128", 128, GREY)]
    xs = [0.85 + i * 1.43 for i in range(len(stages))]
    for x, (label, s, col) in zip(xs, stages):
        h = 0.95 + 2.35 * (s / 128) ** 0.5
        box(ax, x, cy, 1.16, h, label, col, size=8)
    notes = ["3×3 conv s2\n+ 3×3 conv", "", "", "", "1×1 conv\n(bottleneck)", "1×1 conv", "upsample ×2\n+ 2× 3×3 conv",
             "", "", "3×3 conv\n+ sigmoid"]
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 0.59, cy, xs[i + 1] - 0.59, cy)
        if notes[i]:
            ax.text((xs[i] + xs[i + 1]) / 2, 0.32, notes[i], ha="center", fontsize=7.5, color="#4b5563")
    ax.text(8.0, 4.78, "Task 1: convolutional denoising autoencoder", ha="center", fontsize=11, weight="bold", color=INK)
    ax.text(8.0, 4.48, "No skip connections  •  BatchNorm + ReLU in every convolution block  •  dropout on the latent",
            ha="center", fontsize=8.5, color="#4b5563")
    save(fig, "diagram_task1_autoencoder.png")


def task3():
    fig, ax = canvas(14, 6.3)
    box(ax, 1.0, 3.15, 1.65, 1.0, "Corrupted input\nx̃ (3×128×128)", GREY, bold=True)
    box(ax, 4.0, 5.2, 2.85, 0.9, "Gate G (CNN)\ninitialized from Task 2 classifier", PURPLE, size=8.5)
    box(ax, 6.9, 5.2, 2.2, 0.9, "w = softmax(G(x̃) / T)\nT = 2.63", PURPLE)
    branches = [("Identity\n(returns x̃)", GREY), ("Salt-and-pepper\nexpert AE", BLUE), ("Blur\nexpert AE", BLUE),
                ("Occlusion\nexpert AE", BLUE)]
    ys = [4.15, 3.1, 2.05, 1.0]
    for (label, col), y in zip(branches, ys):
        box(ax, 4.65, y - 0.35, 2.3, 0.78, label, col, size=8.5)
        arrow(ax, 1.83, 3.15, 3.45, y - 0.35)
        box(ax, 8.25, y - 0.35, 0.82, 0.62, f"× w{ys.index(y)}", ORANGE, size=9)
        arrow(ax, 5.8, y - 0.35, 7.84, y - 0.35)
        arrow(ax, 8.66, y - 0.35, 9.8, 2.05)
    arrow(ax, 1.83, 3.4, 2.75, 5.2, rad=-0.2)
    arrow(ax, 5.25, 5.2, 5.75, 5.2)
    arrow(ax, 8.25, 4.72, 8.25, 4.52, color="#7c3aed", lw=1.4)
    ax.text(8.7, 4.62, "one weight per branch", fontsize=8, color="#7c3aed", va="center")
    box(ax, 10.25, 2.05, 0.95, 0.95, "Σ", ORANGE, size=16, bold=True)
    arrow(ax, 10.73, 2.05, 11.45, 2.05)
    box(ax, 12.35, 2.05, 1.55, 1.0, "Restored\nx̂ (3×128×128)", GREEN, bold=True)
    ax.text(7.0, 6.05, "Task 3: soft mixture-of-experts", ha="center", fontsize=11, weight="bold", color=INK)
    ax.text(7.0, 5.78, "x̂ = w0·x̃ + w1·A_salt(x̃) + w2·A_blur(x̃) + w3·A_occ(x̃)",
            ha="center", fontsize=9, color="#4b5563")
    ax.text(11.55, 5.1, "Training\n1) warm-up: experts frozen, gate trains\n2) joint fine-tune with smaller lr\n\nLoss\nλ1·L1 + λs·(1−SSIM) + λc·CE + λb·balance",
            fontsize=8.2, color="#374151", va="top", ha="left",
            bbox=dict(facecolor="#f8fafc", edgecolor="#d1d5db", boxstyle="round,pad=0.35"))
    save(fig, "diagram_task3_soft_moe.png")


def task4():
    fig, ax = canvas(15, 8.2)
    ax.text(7.5, 7.98, "Task 4: style-conditioned pix2pix cGAN", ha="center", fontsize=11.5, weight="bold", color=INK)
    # generator (top): a "U" — encoder goes down, decoder comes back up, skips are horizontal lines
    ax.text(0.2, 7.48, "Generator G(x, s): U-Net", fontsize=9.5, weight="bold", color=INK)
    box(ax, 0.95, 6.75, 1.4, 0.8, "Photo x\n3×128×128", GREY, size=8)
    box(ax, 0.95, 5.15, 1.4, 0.8, "Style s ∈ {1,2,3}\nlearned embedding\n(dim 16)", ORANGE, size=7.5)
    enc = ["64\n64²", "128\n32²", "256\n16²", "512\n8²", "512\n4²", "512\n2²", "512\n1²"]
    step, bw = 0.8, 0.68
    ex = [2.5 + i * step for i in range(7)]
    ey = [6.85 - i * 0.5 for i in range(7)]
    for i in range(7):
        box(ax, ex[i], ey[i], bw, 0.6, enc[i], BLUE, size=7.2)
        if i:
            arrow(ax, ex[i - 1] + bw / 2, ey[i - 1], ex[i] - bw / 2, ey[i])
    arrow(ax, 1.65, 6.75, ex[0] - bw / 2, ey[0])
    arrow(ax, 1.65, 5.3, ex[0] - bw / 2, ey[0] - 0.15, rad=0.25)
    ax.text(1.82, 5.92, "broadcast + concat\nstyle map", fontsize=7, color="#b45309", ha="center")
    ax.text(ex[6] + 0.35, ey[6] - 0.05, "style map\nat bottleneck", ha="left", va="center",
            fontsize=6.8, color="#b45309")
    dec = ["512\n2²", "512\n4²", "512\n8²", "256\n16²", "128\n32²", "64\n64²"]
    dx = [ex[6] + (j + 1) * step for j in range(6)]
    dy = [ey[5 - j] for j in range(6)]                      # same level as the mirrored encoder layer
    for j in range(6):
        box(ax, dx[j], dy[j], bw, 0.6, dec[j], GREEN, size=7.2)
        px, py = (ex[6], ey[6]) if j == 0 else (dx[j - 1], dy[j - 1])
        arrow(ax, px + bw / 2, py, dx[j] - bw / 2, dy[j])
        lx1, lx2 = ex[5 - j] + bw / 2, dx[j] - bw / 2       # skip connection at this level
        ax.plot([lx1, lx2], [dy[j] + 0.2, dy[j] + 0.2], color="#9ca3af", lw=0.9, ls="--", zorder=0)
    box(ax, 13.1, 6.1, 1.3, 0.8, "Sketch ŷ\n1×128×128", GREEN, size=8, bold=True)
    arrow(ax, dx[5] + bw / 2, dy[5], 12.7, 6.1)
    ax.text(7.5, 3.08, "4×4 convs, stride 2  •  BatchNorm  •  LeakyReLU (encoder) / ReLU (decoder)\n"
            "dashed lines = concatenated skip connections  •  dropout in the 3 innermost decoder layers",
            ha="center", fontsize=7.6, color="#4b5563")
    # discriminator (bottom)
    ax.text(0.2, 2.48, "Discriminator D(x, y, s): 70×70 PatchGAN", fontsize=9.5, weight="bold", color=INK)
    box(ax, 1.4, 1.15, 2.3, 1.1, "Photo x  +  sketch\n(real y or generated ŷ)\n+ style map", GREY, size=8)
    layers = ["64\n64²", "128\n32²", "256\n16²", "512\n15²", "1\n14²"]
    lxs = [3.6 + i * 1.6 for i in range(5)]
    for i, (x, t) in enumerate(zip(lxs, layers)):
        box(ax, x, 1.15, 1.0, 0.85, t, PURPLE if i < 4 else ORANGE, size=8)
        arrow(ax, (lxs[i - 1] + 0.5) if i else 2.55, 1.15, x - 0.5, 1.15)
    ax.text(12.4, 1.15, "14×14 grid of\nreal/fake logits\n(one per patch)", fontsize=8.5, color="#374151", va="center")
    ax.text(7.5, 0.18, "L_D = ½[BCE(D(x,y,s),1) + BCE(D(x,ŷ,s),0)]\n"
            "L_G = BCE(D(x,ŷ,s),1) + λ_L1·‖y − ŷ‖₁   (λ_L1 = 152.6)",
            ha="center", fontsize=8.8, color=INK)
    save(fig, "diagram_task4_cgan.png")


def app():
    fig, ax = canvas(14, 4.8)
    ax.text(7, 4.55, "Application architecture", ha="center", fontsize=11.5, weight="bold", color=INK)
    ax.text(7, 4.28, "docker compose up --build  →  http://localhost:8080", ha="center",
            fontsize=8.5, color="#4b5563")
    box(ax, 1.3, 2.2, 2.1, 1.4, "Browser\nReact + Tailwind UI\n(4 workspaces)", GREY, size=8.5)
    ax.add_patch(FancyBboxPatch((3.0, 0.35), 10.6, 3.55, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec="#9ca3af", lw=1, ls="--"))
    ax.text(3.2, 3.65, "Docker Compose", fontsize=8.5, color="#6b7280")
    box(ax, 4.6, 2.2, 2.4, 1.4, "frontend container\nnginx :80 (host 8080)\nstatic React build", BLUE, size=8.5)
    box(ax, 8.1, 2.2, 2.6, 1.4, "backend container\nFastAPI + ONNX Runtime\n(CPU, no PyTorch)", GREEN, size=8.5)
    box(ax, 11.95, 2.2, 2.6, 1.6, "models/onnx (read-only)\nuniversal AE, classifier,\n3 experts, soft MoE,\nsketch generator", ORANGE, size=8)
    arrow(ax, 2.35, 2.35, 3.4, 2.35, text="HTTP")
    arrow(ax, 3.4, 2.0, 2.35, 2.0)
    arrow(ax, 5.8, 2.35, 6.8, 2.35, text="/api proxy")
    arrow(ax, 6.8, 2.0, 5.8, 2.0)
    arrow(ax, 9.4, 2.2, 10.65, 2.2, text="volume mount")
    ax.text(8.1, 0.72, "API: /health  ·  /samples  ·  /corrupt  ·  /restore/{universal,hard,soft}  ·  /sketch",
            ha="center", fontsize=7.8, color="#4b5563")
    save(fig, "diagram_app.png")


if __name__ == "__main__":
    task1()
    task3()
    task4()
    app()
