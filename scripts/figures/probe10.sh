cd /data/doubleBee/doubleBee_terr_spawn
echo "########## hE4 run dir checkpoints"
ls logs/co_rl/doublebee_velocity/tqc/energy_abl/2026-09-06_23-10-57_hE4/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tr '\n' ' '; echo
echo "--- ckpts actually used by abl_seeded hE4 / abl_h hE4"
ls abl_seeded/climb_hE4_h06_*.csv | sed 's/.*_//;s/\.csv//' | sort -n | tr '\n' ' '; echo
ls abl_h/climb_hE4_h06_*.csv | sed 's/.*_//;s/\.csv//' | sort -n | tr '\n' ' '; echo
echo
echo "########## who wrote ckpt_sweep (driver + seed + step height)"
grep -rln "ckpt_sweep" *.sh scripts/paper/*.sh scripts/paper/drivers/*.sh 2>/dev/null
grep -rn -- "--step-height\|--seed\|--episodes\|ckpt_sweep" $(grep -rln "ckpt_sweep" *.sh scripts/paper/*.sh scripts/paper/drivers/*.sh 2>/dev/null | head -2) 2>/dev/null | head -20
echo "--- ckpt_sweep file times"
ls -l --time-style=+%F ckpt_sweep/ | head -3
echo
echo "########## ckpt_sweep recomputed at 6 cm (paper metric)"
python3 - <<'PY'
import csv,glob,re,math,statistics as st
def pool(files,h=0.06):
    per=[]
    for f in files:
        r=list(csv.DictReader(open(f)))
        if not r: continue
        cl=100*sum(1 for x in r if float(x["max_gain_m"])>=h)/len(r)
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r]
        e1=[float(x["energy_J"]) for x in r if h<=float(x["max_gain_m"])<2*h]
        per.append((cl,sum(pw)/len(pw),sum(e1)/len(e1) if e1 else float("nan"),len(r)))
    c=[x[0] for x in per]; e1=[x[2] for x in per if not math.isnan(x[2])]
    return (len(per),per[0][3],sum(c)/len(c),st.pstdev(c)/len(c)**0.5 if len(c)>1 else 0,
            sum(x[1] for x in per)/len(per), sum(e1)/len(e1) if e1 else float("nan"))
for t in ("hE0","hE2","hE4","hE6","hE8"):
    fs=sorted(glob.glob("ckpt_sweep/climb_model_*_%s.csv"%t))
    if not fs: continue
    r=pool(fs)
    print("%-4s n=%2d ckpts (%d eps each)  clears %5.1f +- %4.1f %%  power %4.0f W  E(one step) %5.0f J"%(t,r[0],r[1],r[2],r[3],r[4],r[5]))
print()
print("ckpts in sweep:", " ".join(sorted(set(re.search(r"model_(\d+)_",f).group(1) for f in glob.glob("ckpt_sweep/climb_model_*_hE4.csv")),key=int)))
PY
