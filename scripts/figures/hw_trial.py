"""Move the newest hardware trial into trials_final/ and report it.

Window is POLICY TAKEOVER -> DISARM, taken from the `gated` column, so it is
identical for every trial and never hand-picked. gated=="0" is the policy in
control; everything else is FCU disarmed or the CH6 enable low.
"""
import csv, glob, os, shutil, sys
import numpy as np

HW = "/home/airlab/doublebee_PID_JAI/hw_final"
FIN = os.path.join(HW, "trials_final")
SIM_W = 293.0


def segments(g):
    """All contiguous spans where the policy was in control."""
    m = (g == "0").astype(int)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def stats(path, seg=-1):
    r = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(x[k]) for x in r])
    t, z, px, py = f("t"), f("pos_z"), f("pos_x"), f("pos_y")
    qx, qy, qz, qw = f("qx"), f("qy"), f("qz"), f("qw")
    pitch = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    W, E = f("watts"), f("energy_J")
    g = np.array([x["gated"] for x in r])
    segs = segments(g)
    segs = [s for s in segs if s[1] - s[0] >= 10]
    if not segs:
        return None
    # A log can hold several runs. seg picks which; -1 is the last, which is
    # the usual case when an earlier attempt was aborted and re-flown.
    i0, i1_dis = segs[seg]
    # CLIMB WINDOW: takeover -> the moment of maximum height. Disarm timing is
    # operator reaction and varies run to run, so ending on disarm mixes the
    # climb with an arbitrary amount of descent. Ending on peak height makes
    # every trial measure the same physical event.
    i1 = i0 + int(np.argmax(z[i0:i1_dis+1]))
    if i1 - i0 < 10:                      # degenerate, fall back to disarm
        i1 = i1_dis
    m = np.zeros(len(r), bool); m[i0:i1+1] = True
    nseg = len(segs)
    # What happened between peak and disarm, so the window can never be
    # mistaken for concealing a fall.
    z_dis = z[i1_dis]
    hold_s = t[i1_dis] - t[i1]
    dt = np.gradient(t)
    sp = np.convolve(np.hypot(np.gradient(px)/dt, np.gradient(py)/dt),
                     np.ones(9)/9, mode="same")
    up = np.cos(np.radians(np.abs(pitch[m])))
    return dict(
        name=os.path.basename(path), t0=t[i0], t1=t[i1], dur=t[i1]-t[i0],
        n=int(m.sum()), gain=z[i1]-z[i0], peak=z[m].max()-z[i0],
        travel=float(np.hypot(px[m]-px[i0], py[m]-py[i0]).max()),
        vmean=sp[m].mean(), vmax=sp[m].max(),
        E=E[i1]-E[i0], W=W[m].mean(), Wmed=float(np.median(W[m])), Wpk=W[m].max(),
        pmin=pitch[m].min(), pmax=pitch[m].max(), pmean=pitch[m].mean(),
        pdis=pitch[i1], up=up.mean(), upfrac=100*(up > 0.85).mean(),
        nseg=nseg, z_dis=z_dis - z[i0], hold_s=hold_s, t_dis=t[i1_dis])


def main():
    os.makedirs(FIN, exist_ok=True)
    cands = sorted(glob.glob(os.path.join(HW, "trial_*.csv")), key=os.path.getmtime)
    if not cands:
        print("no new trial_*.csv in hw_final/"); return
    src = cands[-1]
    dst = os.path.join(FIN, os.path.basename(src))
    shutil.move(src, dst)
    d = stats(dst)
    if d is None:
        print("MOVED %s -- but the policy was never ungated, no window to report"
              % os.path.basename(dst)); return
    print("=== %s   (moved to trials_final/) ===" % d["name"])
    if d["nseg"] > 1:
        print("  NOTE   %d runs in this log; reporting the LAST one" % d["nseg"])
    print("  window   %.2f -> %.2f s   %.2f s   %d steps  (takeover -> peak height)"
          % (d["t0"], d["t1"], d["dur"], d["n"]))
    print("  GAIN     %.3f m  = %.1f steps of 6 cm" % (d["gain"], d["gain"]/0.06))
    print("  after    held %.1f s to disarm, ending at %+.3f m (%+.3f vs peak)"
          % (d["hold_s"], d["z_dis"], d["z_dis"] - d["gain"]))
    print("  travel   %.2f m   speed mean %.2f  peak %.2f m/s" % (d["travel"], d["vmean"], d["vmax"]))
    print("  energy   %.0f J    power mean %.0f W  median %.0f  peak %.0f" % (d["E"], d["W"], d["Wmed"], d["Wpk"]))
    print("           %.0f%% of sim's %.0f W" % (100*d["W"]/SIM_W, SIM_W))
    print("  pitch    %+.1f .. %+.1f deg  mean %+.1f   at disarm %+.1f" % (d["pmin"], d["pmax"], d["pmean"], d["pdis"]))
    print("  upright  mean %.3f   above 0.85 for %.0f%% of the window" % (d["up"], d["upfrac"]))

    rows = [stats(p) for p in sorted(glob.glob(os.path.join(FIN, "trial_*.csv")))]
    rows = [x for x in rows if x]
    if len(rows) > 1:
        gn = np.array([x["gain"] for x in rows]); wv = np.array([x["W"] for x in rows])
        print()
        print("  --- %d trials in trials_final/ ---" % len(rows))
        print("  gain   %.3f +- %.3f m   (%d of %d cleared >= 1 step)"
              % (gn.mean(), gn.std(ddof=1), int((gn >= 0.06).sum()), len(gn)))
        print("  power  %.0f +- %.0f W" % (wv.mean(), wv.std(ddof=1)))


main()
