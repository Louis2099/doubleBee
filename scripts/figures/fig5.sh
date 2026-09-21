cd /data/doubleBee/doubleBee_terr_spawn
python3 - <<'PY'
import csv, glob, math, statistics as st
H = 0.06
print("Fig. 5 energy ablation at 6 cm, PAIRED SEEDS (abl_seeded), 10 ckpts x 200 eps")
print("%-5s %4s  %-18s %-9s %-12s" % ("wE","n","clears (%)","power (W)","E one step (J)"))
for tag, w in (("hE0","0"),("hE2","2"),("hE4","4"),("hE6","6"),("hE8","8")):
    per = []
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv" % tag)):
        r = list(csv.DictReader(open(f)))
        if not r: continue
        cl = 100*sum(1 for x in r if float(x["max_gain_m"]) >= H)/len(r)
        pw = [float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r]
        e1 = [float(x["energy_J"]) for x in r if H <= float(x["max_gain_m"]) < 2*H]
        per.append((cl, sum(pw)/len(pw), sum(e1)/len(e1) if e1 else float("nan")))
    if not per:
        print("%-5s  ---- missing ----" % w); continue
    c = [p[0] for p in per]
    e1 = [p[2] for p in per if not math.isnan(p[2])]
    se = st.pstdev(c)/len(c)**0.5 if len(c) > 1 else 0
    print("%-5s %4d  %5.1f +- %-8.1f %-9.0f %-12.0f" % (
        w, len(per), sum(c)/len(c), se,
        sum(p[1] for p in per)/len(per),
        sum(e1)/len(e1) if e1 else float("nan")))
print()
print("paper currently says (independent goals): wE0 59+-5, wE4 52+-5, E 764 -> 678 J")
print("Table V learned column (paired seeds)   : 39.8 +- 2.5 at 286 W")
PY
