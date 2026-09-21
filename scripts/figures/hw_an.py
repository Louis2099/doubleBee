import csv, sys
import numpy as np

def analyse(path):
    r = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(x[k]) for x in r])
    t, z, px, py = f("t"), f("pos_z"), f("pos_x"), f("pos_y")
    qx,qy,qz,qw = f("qx"), f("qy"), f("qz"), f("qw")
    pitch = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    W, E = f("watts"), f("energy_J")
    dt = np.gradient(t)
    sp = np.convolve(np.hypot(np.gradient(px)/dt, np.gradient(py)/dt),
                     np.ones(15)/15, mode="same")
    # CLIMB window: contiguous span where power is doing real work (>60 W).
    # Defining it on POWER rather than on a hand-picked time avoids the
    # settling tail, which is what drags the mean down.
    hot = W > 60.0
    if not hot.any():
        return None
    i0, i1 = np.argmax(hot), len(hot) - 1 - np.argmax(hot[::-1])
    m = np.zeros_like(hot); m[i0:i1+1] = True
    pre  = z[max(0,i0-100):i0].mean() if i0 > 10 else z[0]
    post = z[i1:].mean() if i1 < len(z)-10 else z[-1]
    settled = pitch[i1:] if i1 < len(pitch)-10 else pitch[-20:]
    return dict(
        name=path.split("/")[-1], dur=t[i1]-t[i0],
        gain=post-pre, steps=(post-pre)/0.06,
        travel=float(np.hypot(px[m]-px[i0], py[m]-py[i0]).max()),
        vmax=sp[m].max(), Wmean=W[m].mean(), Wpk=W[m].max(),
        E=E[i1]-E[i0], pmin=pitch[m].min(), pmax=pitch[m].max(),
        settled_pitch=settled.mean(),
        upright=np.cos(np.radians(abs(settled.mean()))))

print("%-20s %5s %6s %5s %6s %6s %6s %7s %8s" %
      ("trial","dur","gain","steps","travel","vmax","W_mean","settled","upright"))
for p in sys.argv[1:]:
    d = analyse(p)
    if d is None: print("%-20s  no active window" % p); continue
    print("%-20s %5.1f %6.3f %5.1f %6.2f %6.2f %6.0f %+7.1f %8.3f" %
          (d["name"], d["dur"], d["gain"], d["steps"], d["travel"],
           d["vmax"], d["Wmean"], d["settled_pitch"], d["upright"]))
