cd /data/doubleBee/doubleBee_terr_spawn
python3 - <<'PY'
import csv, glob, re, math, statistics as st
H = 0.06

def by_k(tag):
    """clearance and power keyed by evaluation index k (the shared seed)."""
    out = {}
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_k*.csv" % tag)):
        k = int(re.search(r"_k(\d+)_", f).group(1))
        r = list(csv.DictReader(open(f)))
        if not r: continue
        cl = 100*sum(1 for x in r if float(x["max_gain_m"]) >= H)/len(r)
        pw = sum(float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r)/len(r)
        out[k] = (cl, pw)
    return out

arms = {t: by_k(t) for t in ("hE0","hE2","hE4","hE6","hE8")}
print("per-seed clearance (%), paired across arms\n")
ks = sorted(set.intersection(*[set(v) for v in arms.values()]))
print("k    " + "".join("%8s" % t for t in arms))
for k in ks:
    print("%-5d" % k + "".join("%8.1f" % arms[t][k][0] for t in arms))

print("\npaired differences against hE4 (positive = that arm clears more)\n")
print("%-6s %10s %10s %8s  %s" % ("arm","mean diff","SE","t","power diff (W)"))
base = arms["hE4"]
for t in ("hE0","hE2","hE6","hE8"):
    d = [arms[t][k][0] - base[k][0] for k in ks]
    dp = [arms[t][k][1] - base[k][1] for k in ks]
    m = sum(d)/len(d)
    se = st.stdev(d)/len(d)**0.5 if len(d) > 1 else float("nan")
    print("%-6s %+10.1f %10.1f %8.2f  %+.0f" % (
        t, m, se, m/se if se else float("nan"), sum(dp)/len(dp)))
PY
