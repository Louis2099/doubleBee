cd /data/doubleBee/doubleBee_terr_spawn
echo "########## optionB_driver.sh (Table V)"
cat optionB_driver.sh
echo
echo "########## FIG4 source dir (abl_h)"
ls -d abl_h abl_ckpt 2>/dev/null
ls -1 abl_h/*.csv 2>/dev/null | sed 's|abl_h/||' | head -20
echo "... total $(ls abl_h/*.csv 2>/dev/null | wc -l)"
echo "--- which ckpt dir fed hE4 in abl_h (from any .log)"
grep -ho -- "--checkpoint [^ ]*" abl_h/*.log 2>/dev/null | sort -u | head -8
grep -ho -- "--seed [0-9]*" abl_h/*.log 2>/dev/null | sort -u | head -3
echo
echo "########## FIG5 source (energy ablation)"
for d in energy energy_2 energy_3 energy_abl_res ablations; do echo "--- $d"; ls -1 $d 2>/dev/null | head -12; done
echo
echo "########## RUN DIRS used by each arm"
echo "--- hE4 / main policy checkpoint references across drivers"
grep -ho "logs/co_rl/[^ \"']*" optionB_driver.sh switch_driver4.sh ct_matched_driver.sh 2>/dev/null | sed 's|/model_.*||' | sort | uniq -c
echo
echo "########## eval wall time per run"
python3 - <<'PY'
import os,glob
fs=sorted(glob.glob("actfix_eval/*_h06.log"),key=os.path.getmtime)
ts=[os.path.getmtime(f) for f in fs]
d=[round(ts[i+1]-ts[i]) for i in range(len(ts)-1)]
d=[x for x in d if 0<x<4000]
print("n=%d consecutive gaps, median %ds, min %ds max %ds"%(len(d),sorted(d)[len(d)//2],min(d),max(d)))
PY
