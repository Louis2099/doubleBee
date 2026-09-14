"""Annotation layer for the headline climb composite.

Kept separate from make_overlay.py so the composite can be regenerated without
redoing the labels, and so the labels can be retargeted to a new composite by
editing only the GEOMETRY block.

Terminology: label these "step 1" / "step 2". Never the other word.
"""
import argparse
from PIL import Image, ImageDraw, ImageFont

# ---- GEOMETRY: the only block to edit when the composite changes ------------
GEO = {
    "fig_headline_crop.png": dict(
        poses=[(60, "0.0"), (150, "0.8"), (255, "1.5"), (355, "2.1"),
               (430, "2.6"), (590, "3.4")],
        pose_label_y=292,          # baseline for timestamps
        step1=dict(x=366, y_top=240, y_bot=274, text="6.5 cm"),
        step2=dict(x=624, y_top=206, y_bot=246, text="6 cm"),
        gain=dict(x=792, y_top=206, y_bot=274, text="+0.15 m"),
        callout_prop=dict(xy=(560, 26), target=(598, 72), text="tilting propellers"),
        callout_wheel=dict(xy=(664, 262), target=(606, 218), text="driven wheels"),
    ),
}

FG = (255, 255, 255)
SHADOW = (0, 0, 0)


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def text(d, xy, s, f, anchor="mm"):
    """White text with a 1px dark halo so it reads on carpet or curtain."""
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        d.text((x + dx, y + dy), s, font=f, fill=SHADOW, anchor=anchor)
    d.text(xy, s, font=f, fill=FG, anchor=anchor)


def vdim(d, x, y0, y1, s, f, tick=6):
    """Vertical double-headed dimension arrow with a label to its left."""
    d.line([(x, y0), (x, y1)], fill=FG, width=2)
    for y in (y0, y1):
        d.line([(x - tick, y), (x + tick, y)], fill=FG, width=2)
    for yy, dy in ((y0, 5), (y1, -5)):
        d.line([(x, yy), (x - 4, yy + dy)], fill=FG, width=2)
        d.line([(x, yy), (x + 4, yy + dy)], fill=FG, width=2)
    text(d, (x - 26, (y0 + y1) // 2), s, f)


def leader(d, xy, target, s, f, anchor="lm"):
    d.line([xy, target], fill=FG, width=2)
    d.ellipse([target[0] - 3, target[1] - 3, target[0] + 3, target[1] + 3], fill=FG)
    text(d, xy, s, f, anchor=anchor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--variant", choices=("minimal", "full"), default="full")
    p.add_argument("--drop-first", type=int, default=0,
                   help="crop N leading poses; crops the image left of pose N")
    a = p.parse_args()

    im = Image.open(a.image).convert("RGB")
    key = a.image.split("/")[-1]
    g = GEO[key]
    poses = g["poses"]

    xoff = 0
    if a.drop_first:
        cut = max(0, poses[a.drop_first][0] - 45)
        im = im.crop((cut, 0, im.width, im.height))
        poses = poses[a.drop_first:]
        xoff = cut

    d = ImageDraw.Draw(im)
    fs = max(11, int(im.height * 0.055))
    f = font(fs)
    fsm = font(max(10, fs - 2))

    for x, t in poses:
        text(d, (x - xoff, g["pose_label_y"]), "t = %s s" % t, fsm)

    for k in ("step1", "step2"):
        s = g[k]
        vdim(d, s["x"] - xoff, s["y_top"], s["y_bot"], s["text"], fsm)

    if a.variant == "full":
        gn = g["gain"]
        vdim(d, gn["x"] - xoff, gn["y_top"], gn["y_bot"], gn["text"], fsm)
        for k, anch in (("callout_prop", "mm"), ("callout_wheel", "lm")):
            c = g[k]
            leader(d, (c["xy"][0] - xoff, c["xy"][1]),
                   (c["target"][0] - xoff, c["target"][1]), c["text"], fsm, anch)

    im.save(a.out)
    print("wrote %s  %dx%d" % (a.out, im.width, im.height))


if __name__ == "__main__":
    main()
