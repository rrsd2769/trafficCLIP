"""Architecture diagram of the tri-modal OptimizedTrafficCLIP model (paper Fig. 1).

Grey boxes are frozen pretrained modules, white boxes are trained. The visual
branch follows src/models/traffic_clip.py: the detail-aware encoder and the
frozen ResNet-50 path run in parallel and are concatenated.

Usage:
    .venv/bin/python scripts/plot_architecture.py
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES_DIR = Path("results/figures")
PAPER_DIR = Path("paper")
FROZEN = "#B0B0B0"
TRAINED = "white"


def box(ax, cx, cy, w, h, text, fill=TRAINED, bold=False, size=8.5):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill, edgecolor="black", linewidth=1.1,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                        linewidth=1.0, color="black", shrinkA=0, shrinkB=0)
    )


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    h = 0.75
    box(ax, 6, 10.4, 6.0, 0.7, "Encrypted flow (5-tuple session)", bold=True, size=9.5)

    xv, xp, xs = 2.3, 6.6, 10.0
    for x, name in ((xv, "Visual branch"), (xp, "Physics branch"), (xs, "Semantic branch")):
        ax.text(x - 0.85, 9.3, name, ha="left", va="center", fontsize=9, fontweight="bold")

    # Visual branch: image, then two parallel paths, then concatenation
    box(ax, xv, 8.55, 3.6, h, "28$\\times$28 byte-flow image\n(first 54 bytes zeroed)")
    xa, xb = 1.25, 3.35
    box(ax, xa, 6.9, 1.9, 1.2, "Detail-aware\nCNN encoder")
    box(ax, xb, 7.25, 1.9, h, "ResNet-50", fill=FROZEN)
    box(ax, xb, 6.2, 1.9, h, "Projection +\nresidual adapter")
    box(ax, xv, 4.7, 3.6, h, "Concatenate\n$\\rightarrow$ visual embedding $v$")

    # Physics branch
    box(ax, xp, 8.55, 2.7, h, "Mean IAT, jitter,\nbyte entropy")
    box(ax, xp, 6.9, 2.7, h, "Stats projection\n(MLP)")
    box(ax, xp, 4.7, 2.7, h, "Stats embedding $s$")

    # Semantic branch
    box(ax, xs, 8.55, 3.0, h, "Physics-conditioned\nprompt, one per class")
    box(ax, xs, 6.9, 3.0, h, "BERT text encoder", fill=FROZEN)
    box(ax, xs, 4.7, 3.0, h, "Text embeddings $t_1..t_K$")

    box(ax, 6, 2.75, 6.4, 0.85, "Non-linear MLP fusion head  $f(v, s, t_k)$", bold=True, size=9.5)
    box(ax, 6, 1.2, 6.4, 0.75, "Predicted class: $\\arg\\max_k f(v, s, t_k)$", bold=True, size=9.5)

    bus = 9.7
    ax.plot([6, 6], [10.4 - 0.35, bus], color="black", linewidth=1.0)
    ax.plot([xv - 1.0, xs - 1.0], [bus, bus], color="black", linewidth=1.0)
    for x in (xv, xp, xs):
        arrow(ax, (x - 1.0, bus), (x - 1.0, 8.55 + h / 2))
    # physics statistics also condition the prompt
    arrow(ax, (xp + 1.35, 8.55), (xs - 1.5, 8.55))

    arrow(ax, (xa, 8.55 - h / 2), (xa, 6.9 + 0.6))
    arrow(ax, (xb, 8.55 - h / 2), (xb, 7.25 + h / 2))
    arrow(ax, (xb, 7.25 - h / 2), (xb, 6.2 + h / 2))
    arrow(ax, (xa, 6.9 - 0.6), (xa, 4.7 + h / 2))
    arrow(ax, (xb, 6.2 - h / 2), (xb, 4.7 + h / 2))

    for x in (xp, xs):
        arrow(ax, (x, 8.55 - h / 2), (x, 6.9 + h / 2))
        arrow(ax, (x, 6.9 - h / 2), (x, 4.7 + h / 2))

    for x, xe in ((xv, 3.9), (xp, 6.0), (xs, 8.1)):
        arrow(ax, (x, 4.7 - h / 2), (xe, 2.75 + 0.425))
    arrow(ax, (6, 2.75 - 0.425), (6, 1.2 + 0.375))

    ax.add_patch(plt.Rectangle((0.3, 0.2), 0.35, 0.3, facecolor=FROZEN, edgecolor="black"))
    ax.text(0.8, 0.35, "Frozen pretrained", va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((3.3, 0.2), 0.35, 0.3, facecolor=TRAINED, edgecolor="black"))
    ax.text(3.8, 0.35, "Trained", va="center", fontsize=8)

    fig.tight_layout(pad=0.2)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "architecture_bw.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    shutil.copy(out_path, PAPER_DIR / out_path.name)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
