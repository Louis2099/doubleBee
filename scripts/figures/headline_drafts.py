"""Four annotation drafts for the Fig. 1 headline composite (2600x1233)."""
from PIL import Image, ImageDraw, ImageFont
import os

SRC = "/home/airlab/Downloads/figs_fixed/headline_compressed.jpg"
OUT = "/tmp/claude-1000/-home-airlab-doublebee-PID-JAI/4a1ea8f7-460d-4372-9a50-1f738c3051e4/scratchpad/headline_drafts"
os.makedirs(OUT, exist_ok=True)
S = 1.3  # coordinates below are measured on the 2000 px preview

FG, SH = (255, 255, 255), (0, 0, 0)
ACC = (255, 196, 0)


def P(x, y):
    return (int(x * S), int(y * S))


def font(sz, bold=True):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    return ImageFont.truetype(p, sz)


def text(d, xy, s, f, anchor="mm", fill=FG, halo=4):
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=halo, stroke_fill=SH)


def line(d, pts, fill=FG, w=6):
    d.line(pts, fill=SH, width=w + 6)
    d.line(pts, fill=fill, width=w)


def vdim(d, x, y0, y1, s, f, side="left", fill=FG):
    x, y0 = P(x, y0)[0], P(x, y0)[1]
    y1 = int(y1 * S)
    t = 22
    line(d, [(x, y0), (x, y1)], fill)
    line(d, [(x - t, y0), (x + t, y0)], fill)
    line(d, [(x - t, y1), (x + t, y1)], fill)
    dx = -40 if side == "left" else 40
    text(d, (x + dx, (y0 + y1) // 2), s, f, anchor="rm" if side == "left" else "lm", fill=fill)


def arrow(d, x0, y, x1, fill=FG, w=8):
    a, b = P(x0, y), P(x1, y)
    line(d, [a, b], fill, w)
    h = 34
    tri = [b, (b[0] - h, b[1] - h // 1.6), (b[0] - h, b[1] + h // 1.6)]
    d.polygon([(p[0], p[1]) for p in tri], fill=fill, outline=SH)


def leader(d, frm, to, s, f, anchor="lm", fill=FG):
    a, b = P(*frm), P(*to)
    line(d, [a, b], fill, 5)
    r = 12
    d.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], fill=fill, outline=SH, width=3)
    off = 14 if anchor[0] == "l" else -14
    text(d, (a[0] + off, a[1]), s, f, anchor=anchor, fill=fill)


def badge(d, cx, cy, s, f, r=48, fill=FG):
    c = P(cx, cy)
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(0, 0, 0), outline=fill, width=6)
    d.text(c, s, font=f, fill=fill, anchor="mm")


base = Image.open(SRC).convert("RGB")
F = font(66)
Fs = font(56)

# Step front faces measured on the preview.
STEP1 = dict(x=690, y0=848, y1=938)
STEP2 = dict(x=1185, y0=782, y1=848)

# ---- A: heights + direction only -------------------------------------------
im = base.copy(); d = ImageDraw.Draw(im)
vdim(d, STEP1["x"], STEP1["y0"], STEP1["y1"], "6 cm", F)
vdim(d, STEP2["x"], STEP2["y0"], STEP2["y1"], "6 cm", F)
arrow(d, 60, 905, 400)
im.save(f"{OUT}/A_heights_arrow.jpg", quality=92)

# ---- B: numbered exposures + step labels ------------------------------------
im = base.copy(); d = ImageDraw.Draw(im)
for i, (x, y) in enumerate([(250, 120), (690, 100), (980, 100), (1420, 40)], 1):
    badge(d, x, y, str(i), Fs)
text(d, P(960, 905), "step 1", F)
text(d, P(1600, 815), "step 2", F)
vdim(d, STEP1["x"], STEP1["y0"], STEP1["y1"], "6 cm", F)
vdim(d, STEP2["x"], STEP2["y0"], STEP2["y1"], "6 cm", F)
im.save(f"{OUT}/B_numbered_steps.jpg", quality=92)

# ---- C: hardware callouts on the final pose + heights ------------------------
im = base.copy(); d = ImageDraw.Draw(im)
leader(d, (1300, 130), (1480, 85), "tilting propellers", Fs, anchor="rm")
leader(d, (1700, 470), (1590, 520), "tilt servos", Fs) if False else None
leader(d, (1360, 720), (1470, 600), "driven wheels", Fs, anchor="rm")
leader(d, (1300, 250), (1250, 330), "tail", Fs, anchor="rm") if False else None
vdim(d, STEP1["x"], STEP1["y0"], STEP1["y1"], "6 cm", F)
vdim(d, STEP2["x"], STEP2["y0"], STEP2["y1"], "6 cm", F)
arrow(d, 60, 905, 400)
im.save(f"{OUT}/C_hardware_callouts.jpg", quality=92)

# ---- D: accent colour, heights + total gain on the right --------------------
im = base.copy(); d = ImageDraw.Draw(im)
vdim(d, STEP1["x"], STEP1["y0"], STEP1["y1"], "6 cm", F, fill=ACC)
vdim(d, STEP2["x"], STEP2["y0"], STEP2["y1"], "6 cm", F, fill=ACC)
vdim(d, 1960, 782, 938, "+12 cm", F, side="left", fill=ACC)
arrow(d, 60, 905, 400, fill=ACC)
im.save(f"{OUT}/D_accent_gain.jpg", quality=92)

# contact sheet for review
thumbs = [Image.open(f"{OUT}/{n}") for n in sorted(os.listdir(OUT)) if n[0] in "ABCD"]
w, h = 1300, int(1233 * 1300 / 2600)
sheet = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
for k, t in enumerate(thumbs):
    sheet.paste(t.resize((w, h)), ((k % 2) * w, (k // 2) * h))
sheet.save(f"{OUT}/sheet.jpg", quality=85)
print("ok")
