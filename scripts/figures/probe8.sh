cd /data/doubleBee/doubleBee_terr_spawn
echo "########## where do hE0/hE2/hE6/hE8 eval CSVs live?"
find . -maxdepth 2 -name "*hE[0-9]*.csv" 2>/dev/null | sed 's|/[^/]*$||' | sort | uniq -c
echo "--- tags per candidate dir"
for d in abl_ckpt climb_eval abl_eval_play abl_eval_play_rep2 pid_eval; do
  echo "--- $d ($(ls $d/*.csv 2>/dev/null | wc -l) csv)"
  ls -1 $d/*.csv 2>/dev/null | sed "s|$d/climb_||;s|_h[0-9]*||;s|_[0-9]*\.csv||" | sort | uniq -c | head -12
done
echo
echo "########## energy-weight ablation, 6 cm, recomputed"
python3 - <<'PY'
import csv,glob,os,statistics as st
def pool(files,h=0.06):
    per=[]; one=[]
    for f in files:
        r=list(csv.DictReader(open(f)))
        if not r: continue
        cl=100*sum(1 for x in r if float(x["max_gain_m"])>=h)/len(r)
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r]
        # energy on episodes clearing exactly one step (gain in [h, 2h))
        e1=[float(x["energy_J"]) for x in r if h<=float(x["max_gain_m"])<2*h]
        per.append((cl,sum(pw)/len(pw),sum(e1)/len(e1) if e1 else float("nan"),len(e1)))
    if not per: return None
    c=[x[0] for x in per]
    import math
    e1=[x[2] for x in per if not math.isnan(x[2])]
    return (len(per),sum(c)/len(c),st.pstdev(c)/len(c)**0.5 if len(c)>1 else 0,
            sum(x[1] for x in per)/len(per), sum(e1)/len(e1) if e1 else float("nan"))
for d in ("abl_ckpt","abl_h","climb_eval","abl_eval_play","abl_eval_play_rep2"):
    if not os.path.isdir(d): continue
    tags=sorted(set(os.path.basename(f).split("climb_")[-1].split("_h0")[0] for f in glob.glob(d+"/climb_*h0*.csv")))
    tags=[t for t in tags if t.startswith(("hE","gE","wE"))]
    if not tags: continue
    print("=== %s"%d)
    for t in tags:
        r=pool(sorted(glob.glob("%s/climb_%s_h06*.csv"%(d,t))))
        if r: print("  %-6s n=%2d clears %5.1f +- %4.1f %%  power %4.0f W  E(one step) %5.0f J"%(t,r[0],r[1],r[2],r[3],r[4]))
PY
