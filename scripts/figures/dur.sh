cd /data/doubleBee/doubleBee_terr_spawn
python3 - <<'PY'
import csv, glob
NAMES={"hE4":"Learned (w_E=4)","ct10":"Fixed T/W 0.55","ct050":"Fixed T/W 0.46",
       "ctm05":"Fixed T/W 0.31","ctm45":"Fixed T/W 0.23",
       "swA3":"Switch 0.31/0.55","swB3":"Switch 0.46/0.55"}
H=0.06
print("Why conditional energy differs: EPISODE DURATION, not power\n")
print("%-18s %10s %10s %10s %10s"%("arm","dur_all(s)","dur_1step(s)","E1step(J)","P(W)"))
for tag in ("hE4","swA3","swB3","ct10","ct050","ctm05","ctm45"):
    da=[];d1=[];e1=[];pw=[]
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv"%tag)):
        for x in csv.DictReader(open(f)):
            s=int(x["steps"]); g=float(x["max_gain_m"]); e=float(x["energy_J"])
            da.append(s*0.02); pw.append(e/max(1,s)/0.02)
            if H<=g<2*H: d1.append(s*0.02); e1.append(e)
    if not da: continue
    print("%-18s %10.2f %10s %10s %10.0f"%(
        NAMES[tag],sum(da)/len(da),
        "%.2f"%(sum(d1)/len(d1)) if d1 else "--",
        "%.0f"%(sum(e1)/len(e1)) if e1 else "--",
        sum(pw)/len(pw)))
PY
