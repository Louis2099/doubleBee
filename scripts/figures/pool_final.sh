cd /data/doubleBee/doubleBee_terr_spawn
python3 - <<'PY'
import csv, glob, math, statistics as st
H=0.06
NAMES={"hE4":"Learned (w_E=4)","ct10":"Fixed T/W 0.55","ct050":"Fixed T/W 0.46",
       "ctm05":"Fixed T/W 0.31","ctm45":"Fixed T/W 0.23",
       "swA3":"Switch 0.31/0.55","swB3":"Switch 0.46/0.55"}
print("6 cm, PAIRED SEEDS, 10 ckpts x 200 episodes -- FINAL\n")
print("%-18s %3s %-16s %-9s %-10s %s"%("arm","n","clears (%)","power (W)","E1step(J)","n_1step"))
pts=[]
for tag in ("hE4","swA3","swB3","ct10","ct050","ctm05","ctm45"):
    per=[];n1=0
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv"%tag)):
        rr=list(csv.DictReader(open(f)))
        if not rr: continue
        cl=100*sum(1 for x in rr if float(x["max_gain_m"])>=H)/len(rr)
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in rr]
        e1=[float(x["energy_J"]) for x in rr if H<=float(x["max_gain_m"])<2*H]
        n1+=len(e1)
        per.append((cl,sum(pw)/len(pw),sum(e1)/len(e1) if e1 else float("nan")))
    if not per: continue
    c=[p[0] for p in per]; ee=[p[2] for p in per if not math.isnan(p[2])]
    se=st.pstdev(c)/len(c)**0.5 if len(c)>1 else 0
    print("%-18s %3d %5.1f +- %-8.1f %-9.0f %-10s %d"%(
        NAMES[tag],len(per),sum(c)/len(c),se,sum(p[1] for p in per)/len(per),
        "%.0f"%(sum(ee)/len(ee)) if ee else "--", n1))
    pts.append((NAMES[tag],sum(c)/len(c),sum(p[1] for p in per)/len(per)))
print("\nFig. 4 bottom panel, clearance vs mean power:")
for n,c,p in sorted(pts,key=lambda q:-q[2]):
    print("   %-18s %5.1f %%   %4.0f W"%(n,c,p))
PY
