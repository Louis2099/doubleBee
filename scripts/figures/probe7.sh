cd /data/doubleBee/doubleBee_terr_spawn
echo "########## energy-ablation run dirs"
ls -d logs/co_rl/doublebee_velocity/tqc/energy_abl/* 2>/dev/null
echo "--- fixed/const-thrust run dirs"
ls -d logs/co_rl/doublebee_velocity/tqc/*abl_ct* logs/co_rl/doublebee_velocity/tqc/*abl_sw* 2>/dev/null
echo
echo "########## scripts/paper energy dirs"
for d in energy energy_2 energy_3 energy_abl_res ablations results cmds drivers; do
  echo "--- scripts/paper/$d"; ls -1 scripts/paper/$d 2>/dev/null | head -10
done
echo
echo "########## abl_h arm tags"
ls -1 abl_h/*.csv | sed 's|abl_h/climb_||;s|_h[0-9]*_[0-9]*\.csv||' | sort | uniq -c
echo
echo "########## abl_seeded status"
ls -1 abl_seeded/*.csv 2>/dev/null | sed 's|abl_seeded/climb_||;s|_h\([0-9]*\)_.*|  h\1|' | sort | uniq -c
echo
echo "########## 6 cm CLEARANCE recomputed, paper metric (max_gain >= 0.06)"
python3 - <<'PY'
import csv,glob,re,collections,statistics as st
def pool(files):
    per=[]
    for f in files:
        r=list(csv.DictReader(open(f)))
        if not r: continue
        cl=100*sum(1 for x in r if float(x["max_gain_m"])>=0.06)/len(r)
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r]
        per.append((cl,sum(pw)/len(pw)))
    if not per: return None
    c=[x[0] for x in per]; p=[x[1] for x in per]
    return len(per),sum(c)/len(c),(st.pstdev(c)/len(c)**0.5 if len(c)>1 else 0),sum(p)/len(p)
print("=== abl_h  (Fig. 4 / Fig. 5 source, unseeded)")
tags=sorted(set(re.sub(r"_h\d+_\d+\.csv$","",f.split("climb_")[1]) for f in glob.glob("abl_h/*.csv")))
for t in tags:
    r=pool(sorted(glob.glob("abl_h/climb_%s_h06_*.csv"%t)))
    if r: print("  %-10s n=%2d  clears %5.1f +- %4.1f %%   power %4.0f W"%(t,r[0],r[1],r[2],r[3]))
print("=== abl_seeded  (Table V, paired seeds)")
tags=sorted(set(re.sub(r"_h\d+_k\d+_\d+\.csv$","",f.split("climb_")[1]) for f in glob.glob("abl_seeded/*.csv")))
for t in tags:
    r=pool(sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv"%t)))
    if r: print("  %-10s n=%2d  clears %5.1f +- %4.1f %%   power %4.0f W"%(t,r[0],r[1],r[2],r[3]))
PY
