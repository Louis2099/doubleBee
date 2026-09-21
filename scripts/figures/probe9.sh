cd /data/doubleBee/doubleBee_terr_spawn
echo "########## root-level hE / climb csvs"
ls -1 *hE*.csv climb_*.csv 2>/dev/null | head -20
echo
echo "########## tags in remaining dirs"
for d in ckpt_sweep abl_servo10 sweep_0908_0057 climb_single eval_0908_0401; do
  echo "--- $d ($(ls $d/*.csv 2>/dev/null | wc -l) csv)"
  ls -1 $d/*.csv 2>/dev/null | sed "s|$d/||;s|climb_||;s|_[0-9]\{3,4\}\.csv||;s|\.csv||" | sort | uniq -c | head -14
done
echo
echo "########## every dir holding hE0 or hE8"
find . -maxdepth 2 \( -name "*hE0*" -o -name "*hE8*" -o -name "*hE2*" -o -name "*hE6*" \) -name "*.csv" | head -20
echo
echo "########## RUNBOOK / notes mentioning the energy figure"
grep -rn -i "fig. 5\|fig5\|energy_weight\|energy ablation\|wE = 4\|hE0" scripts/paper/RUNBOOK_switched_thrust.md scripts/paper/frozen_config.txt 2>/dev/null | head -20
echo
echo "########## recompute hE arms at 6 cm wherever they live"
python3 - <<'PY'
import csv,glob,os,math,statistics as st
def pool(files,h=0.06):
    per=[]
    for f in files:
        r=list(csv.DictReader(open(f)))
        if not r: continue
        cl=100*sum(1 for x in r if float(x["max_gain_m"])>=h)/len(r)
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r]
        e1=[float(x["energy_J"]) for x in r if h<=float(x["max_gain_m"])<2*h]
        per.append((cl,sum(pw)/len(pw),sum(e1)/len(e1) if e1 else float("nan")))
    if not per: return None
    c=[x[0] for x in per]; e1=[x[2] for x in per if not math.isnan(x[2])]
    return (len(per),sum(c)/len(c),st.pstdev(c)/len(c)**0.5 if len(c)>1 else 0,
            sum(x[1] for x in per)/len(per), sum(e1)/len(e1) if e1 else float("nan"))
seen={}
for f in glob.glob("*/*.csv")+glob.glob("*.csv"):
    b=os.path.basename(f)
    for t in ("hE0","hE2","hE4","hE6","hE8"):
        if t in b and ("h06" in b or "h6" in b or "_h" not in b):
            seen.setdefault((os.path.dirname(f) or ".",t),[]).append(f)
for (d,t) in sorted(seen):
    r=pool(sorted(seen[(d,t)]))
    if r: print("%-16s %-4s n=%2d clears %5.1f +- %4.1f %%  power %4.0f W  E(one step) %5.0f J"%(d,t,r[0],r[1],r[2],r[3],r[4]))
PY
