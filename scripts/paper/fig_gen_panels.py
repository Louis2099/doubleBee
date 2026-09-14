"""Generalization figure: render on top, base-link traces underneath.

Three terrains the policy never trained on, one column each, laid out after the
brachiation paper's Fig 7: a render with a Greek badge on top, then horizontal
progress and height against time.

The trace is env 0's best episode from eval_climb_gen.py --traj_out, and the
SAME checkpoint is used for every column. That matters: the claim is that one
policy handles all three, so picking a different checkpoint per panel would
quietly weaken it into "some policy in our family handles each case".
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

PANELS = [
    (r"$\alpha$", "NarrowTread", "GenNarrow",
     "Narrow", "0.20 m, trained on 0.40"),
    (r"$\beta$", "RampUp", "GenSlopeUp",
     "Ramp up", "continuous incline, no steps"),
    (r"$\gamma$", "StepDown", "GenStairDown",
     "Step down", "trained on ascent only"),
]
# Distance was the top row and was dropped: all three terrains produce the
# same monotone ramp, so it spent a third of the figure conveying nothing that
# differs between panels. Pitch separates them and shows the control response.
# Taken from the same viridis ramp fig_energy_tradeoff.py uses, so the
# generalization figure reads as part of the same set rather than a visitor.
C_PITCH = plt.cm.viridis(0.28)   # blue
C_Z = plt.cm.viridis(0.68)       # green


def load_traj(d, tag, ckpt):
    p = os.path.join(d, "traj_%s_%s.csv" % (tag, ckpt))
    if not os.path.exists(p):
        return None
    r = list(csv.DictReader(open(p)))
    f = lambda k: np.array([float(x[k]) for x in r])
    t, x, y, z, pitch = f("t"), f("x"), f("y"), f("z"), f("pitch_deg")
    d_ = np.hypot(x, y)
    # trailing post-reset samples snap back toward the spawn; trim them
    keep = len(d_)
    while keep > 2 and d_[keep - 1] < 0.5 * d_[: keep - 1].max():
        keep -= 1
    s = slice(0, keep)
    return t[s], d_[s], z[s], pitch[s]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="figs/gen")
    p.add_argument("--traj_dir", default="gen_final_traj")
    p.add_argument("--ckpt", default="5000")
    p.add_argument("-o", "--out", default="figs/fig_generalization.pdf")
    p.add_argument("--width", type=float, default=7.16)
    p.add_argument("--height", type=float, default=None,
                   help="default 2.35 with the pitch row, 2.2 without")
    p.add_argument("--no_pitch", action="store_true",
                   help="drop the pitch row. Saves a third of the figure; the "
                        "two numbers it carries (3.9 deg on the ramp against "
                        "14.1 on narrow tread) fit in one sentence of text.")
    p.add_argument("--trace_lw", type=float, default=0.8,
                   help="trace weight. Matched to the error-bar and front weight "
                        "in the energy and actuator figures.")
    p.add_argument("--lab", type=float, default=6.0, help="axis label size")
    p.add_argument("--tick", type=float, default=5.5, help="tick label size")
    p.add_argument("--ylabel_x", type=float, default=-0.205,
                   help="axes-fraction x for both y labels, so they align. "
                        "Must satisfy |ylabel_x| * axes_width <= label_pad or "
                        "the label spills into the previous column.")
    p.add_argument("--label_pad", type=float, default=0.060,
                   help="figure-fraction inset applied to the plot axes so "
                        "their labels sit inside the render's width")
    p.add_argument("--crop", default="0.06,0.94",
                   help="vertical fraction of each render to keep. The shots "
                        "carry a lot of empty sky and foreground, and an "
                        "uncropped 1100x723 leaves a void under the image "
                        "because imshow preserves aspect inside a wide cell.")
    a = p.parse_args()

    # Smaller labels are what let the columns close up: narrower tick text
    # means the y label can sit closer to the axis, which means a smaller
    # inset, which means less gutter. The figure is placed at its rendered
    # width, so these are the sizes that print.
    # 6.0/5.5 is deliberate: it is what lets the three columns close up. Raising
    # it to match the other figures collides the y-labels with the neighbouring
    # panel. --lab/--tick expose it if the padding is retuned too.
    plt.rcParams.update({"font.size": 7.0, "axes.labelsize": a.lab,
                         "xtick.labelsize": a.tick, "ytick.labelsize": a.tick,
                         "axes.linewidth": 0.7})
    nrow = 2 if a.no_pitch else 3
    fig = plt.figure(figsize=(a.width,
                              a.height or (2.2 if a.no_pitch else 2.35)))
    lo, hi = (float(v) for v in a.crop.split(","))
    # Row 0's height must MATCH the cropped image aspect or imshow centres the
    # picture in a taller cell and leaves a void underneath. With a 1100 x ~640
    # crop in a ~2.1 in wide column the render is ~1.2 in tall, which is what
    # the ratio below reserves.
    # The render dominates and the traces are short strips beneath it, as in
    # the reference layout. Plots taller than the render, which is what equal
    # ratios give, spend most of a column's height on two monotone curves.
    # Row 0's ratio is chosen so the CELL matches the render's aspect. Getting
    # this wrong is not cosmetic: too tall a row and the aspect-matching crop
    # below trims the sides instead, which zooms in horizontally and reads as a
    # stretched picture. Size the row to the image, do not crop the image to
    # the row.
    # Row 0's ratio still tracks the render aspect; only the trace rows are
    # compressed. Shrinking the figure without keeping that ratio would push
    # the aspect-matching crop into trimming the sides of each render.
    gs = GridSpec(nrow, 3, figure=fig,
                  # The render row is effectively floor-bound: at this width a
                  # cell is ~2.3 in across and the crop's 1.73 aspect fixes it
                  # near 1.3 in. All compression therefore comes out of the
                  # trace rows, which is why their ratio is so much smaller.
                  height_ratios=([1.45, 0.7] if a.no_pitch else [1.45, 0.42, 0.42]),
                  hspace=0.10, wspace=0.045)
    pitch_axes = []
    plot_axes = []

    # `sub` is retained in PANELS as a record of what each terrain changes,
    # but is not drawn: the caption carries it and two lines of overlay text
    # competed with the render.
    for j, (greek, img, tag, title, sub) in enumerate(PANELS):
        # ---- render -------------------------------------------------------
        ax = fig.add_subplot(gs[0, j])
        path = None
        for ext in (".jpeg", ".jpg", ".png"):
            q = os.path.join(a.img_dir, img + ext)
            if os.path.exists(q):
                path = q
                break
        if path:
            arr = np.asarray(Image.open(path))
            h0 = arr.shape[0]
            arr = arr[int(lo * h0):int(hi * h0)]          # trim sky/foreground
            # Then CROP to the cell's aspect rather than stretching to it.
            # aspect="auto" alone fills the cell by distorting the picture: a
            # 1.73 crop in a 1.43 cell squeezes the robots horizontally. Taking
            # a centre crop at the cell's own aspect means "auto" has nothing
            # left to stretch, so the render fills the width edge to edge AND
            # the geometry stays true.
            pos = ax.get_position()
            cell = ((pos.width * fig.get_figwidth())
                    / (pos.height * fig.get_figheight()))
            ih, iw = arr.shape[:2]
            if iw / ih > cell:                 # too wide: trim the sides
                keep_w = int(round(ih * cell))
                x0 = (iw - keep_w) // 2
                arr = arr[:, x0:x0 + keep_w]
            else:                              # too tall: trim top and bottom
                keep_h = int(round(iw / cell))
                y0 = (ih - keep_h) // 2
                arr = arr[y0:y0 + keep_h]
            ax.imshow(arr, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # badge, then the name beside it
        ax.text(0.022, 0.945, greek, transform=ax.transAxes, fontsize=8.5,
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.25", fc="black", ec="none"))
        ax.text(0.115, 0.925, title, transform=ax.transAxes, fontsize=7.5,
                color="white", va="top", ha="left")

        # ---- traces -------------------------------------------------------
        tr = load_traj(a.traj_dir, tag, a.ckpt)
        axp = None if a.no_pitch else fig.add_subplot(gs[1, j])
        axz = fig.add_subplot(gs[nrow - 1, j],
                              **({} if axp is None else dict(sharex=axp)))
        if tr is not None:
            t, dist, z, pitch = tr
            if axp is not None:
                axp.plot(t, pitch, color=C_PITCH, lw=a.trace_lw)
            axz.plot(t, z, color=C_Z, lw=a.trace_lw)
            axz.axhline(0.0, color="0.75", lw=0.6, ls=":")
        for ax_ in ([axz] if axp is None else [axp, axz]):
            ax_.grid(alpha=0.25, lw=0.5)
            # Short strips cannot carry the default tick density.
            ax_.yaxis.set_major_locator(plt.MaxNLocator(3))
            # Strip trailing zeros: "-0.3" instead of "-0.30" is one character
            # narrower, and the descent panel's negatives are the widest tick
            # labels in the figure.
            ax_.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: ("%g" % v)))
            ax_.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax_.tick_params(pad=1.5)
        # Shares its time axis with the height row, so the tick LABELS are
        # redundant. The spine stays: without it the panel loses its frame.
        if axp is not None:
            plt.setp(axp.get_xticklabels(), visible=False)
            axp.tick_params(axis="x", length=0)
            pitch_axes.append(axp)
            plot_axes.append((axp, axz))
        else:
            plot_axes.append((axz, axz))
        axz.set_xlabel("t (s)")
        # Labelled on EVERY column, as in the reference layout. With a uniform
        # inset applied so each plot block lines up with the render above it,
        # labelling only the first column leaves the other two with an empty
        # margin where their label would be.
        # Axis-style labels, as in the reference. Short labels are what let
        # the columns close up. Height is z, not y: y is lateral in this body
        # frame. Pitch uses the same symbol the System Overview defines.
        if axp is not None:
            axp.set_ylabel("ϑ (deg)")
            axp.yaxis.set_label_coords(a.ylabel_x, 0.5)
        axz.set_ylabel("z (m)")
        # Pin both y labels to the same x. Left to itself matplotlib offsets
        # each by its own tick-label width, so "12" and "-0.30" push their
        # labels to different depths and the two rows look misaligned.
        axz.yaxis.set_label_coords(a.ylabel_x, 0.5)

    # One y-scale across the pitch row. With independent axes the narrow-tread
    # panel peaks at 14 deg and the ramp at 4, but both fill their box, so the
    # reader sees the same shape at three and a half times the magnitude.
    if pitch_axes:
        hi_ = max(ax_.get_ylim()[1] for ax_ in pitch_axes)
        for ax_ in pitch_axes:
            ax_.set_ylim(-0.5, hi_)

    # Inset every plot axis from the left of its cell so that the axis PLUS its
    # y label and tick labels together span the same width as the render above,
    # rather than the labels hanging off to the left of it. Applied uniformly so
    # all three columns stay the same width even though only the first carries a
    # y label.
    for axp_, axz_ in plot_axes:
        for ax_ in (axp_, axz_):
            b = ax_.get_position()
            ax_.set_position([b.x0 + a.label_pad, b.y0,
                              b.width - a.label_pad, b.height])

    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.01, dpi=300)
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.01, dpi=220)
    print("wrote %s and %s" % (a.out, png))
    for _, img, tag, title, _ in PANELS:
        tr = load_traj(a.traj_dir, tag, a.ckpt)
        if tr is None:
            print("  %-16s NO TRAJECTORY for ckpt %s" % (title, a.ckpt))
        else:
            t, d, z, pi = tr
            print("  %-16s %.2f m in %.1f s, dz %+.3f m, pitch %.1f..%.1f deg"
                  % (title, d[-1], t[-1], z[-1] - z[0], pi.min(), pi.max()))


if __name__ == "__main__":
    main()
