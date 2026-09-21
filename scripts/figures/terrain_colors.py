"""Preview candidate terrain colours as they will actually appear.

Two things make a naive swatch misleading:

1. PreviewSurfaceCfg diffuse_color is LINEAR. What you see on screen is
   gamma-encoded, so a linear 0.09 displays around 0.33, not 0.09. Previewing
   the raw tuple makes every candidate look far darker than it renders.
2. The question is not "what colour is it" but "can you still see the ramp".
   That is shading headroom, so each colour is shown across the illumination
   range a 0 to 10 degree incline spans under dome light.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANDS = [
    ("A  current", (0.09, 0.13, 0.22)),
    ("B  one stop up", (0.13, 0.18, 0.28)),
    ("C  blue-grey", (0.16, 0.19, 0.24)),
    ("D  saturated", (0.07, 0.12, 0.26)),
]
ROBOT = (0.72, 0.66, 0.50)      # pale body as it renders
AMBIENT, DIRECT = 0.35, 0.65    # dome light plus key


def srgb(lin):
    return np.clip(lin, 0, 1) ** (1 / 2.2)


fig, axes = plt.subplots(len(CANDS), 3, figsize=(9.5, 5.6),
                         gridspec_kw=dict(width_ratios=[3.4, 1.0, 1.0]))
f = np.linspace(0.55, 1.0, 400)          # surface normal turning from/to light
for row, (name, c) in enumerate(zip([n for n, _ in CANDS], [c for _, c in CANDS])):
    lin = np.array(c)[None, :] * (AMBIENT + DIRECT * f)[:, None]
    strip = srgb(np.tile(lin[None, :, :], (60, 1, 1)))

    ax = axes[row][0]
    ax.imshow(strip, aspect="auto")
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9,
                  labelpad=8)
    # the robot sitting on it, for contrast
    ax.add_patch(plt.Rectangle((150, 12), 90, 36, color=srgb(np.array(ROBOT)),
                               ec="none"))
    ax.text(0.015, 0.5, "shaded", color="white", fontsize=7, va="center",
            transform=ax.transAxes, alpha=0.75)

    axes[row][1].imshow(srgb(np.tile(np.array(c)[None, None, :], (10, 10, 1))))
    axes[row][1].set_xticks([]); axes[row][1].set_yticks([])
    if row == 0:
        axes[row][1].set_title("flat", fontsize=8)

    g = (0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2])
    gs = srgb(np.tile(g[None, :, None], (60, 1, 3)))
    axes[row][2].imshow(gs, aspect="auto")
    axes[row][2].set_xticks([]); axes[row][2].set_yticks([])
    if row == 0:
        axes[row][2].set_title("greyscale print", fontsize=8)

    rng = srgb(lin[-1]) - srgb(lin[0])
    axes[row][0].text(0.985, 0.5,
                      "rgb %.2f %.2f %.2f   shading range %.2f"
                      % (c[0], c[1], c[2], rng.mean()),
                      color="white", fontsize=7, ha="right", va="center",
                      transform=axes[row][0].transAxes, alpha=0.9)

fig.suptitle("Terrain colour candidates: left = same colour across the illumination "
             "a shallow ramp spans", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figs/terrain_color_options.png", dpi=150, bbox_inches="tight")
print("wrote figs/terrain_color_options.png")
for name, c in CANDS:
    lin = np.array(c)
    print("  %-16s linear %s  ->  displays as %s"
          % (name, tuple(round(v, 3) for v in lin),
             tuple(round(v, 2) for v in srgb(lin))))
