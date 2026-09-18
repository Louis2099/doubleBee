#!/bin/bash
# Overnight: (1) pooled seeded evaluation of the matched-actuator runs,
#            (2) matched-actuator switch baseline (rebuttal material).
set -u
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
ACT="DOUBLEBEE_SERVO_VEL_LIMIT=10.0 DOUBLEBEE_WHEEL_ARMATURE=0.0085 DOUBLEBEE_PROP_EFFORT=25 DOUBLEBEE_PROP_DAMPING=0.16 DOUBLEBEE_PROP_ARMATURE=0.0016"
W40=$(ls -dt logs/co_rl/doublebee_velocity/tqc/*_actfix_wE40 | head -1)
W02=$(ls -dt logs/co_rl/doublebee_velocity/tqc/*_actfix_wE025 | head -1)
mkdir -p actfix_eval sweep_logs/actfix
LOG=sweep_logs/actfix/night.log
echo "[night] start $(date -u)" >> $LOG

# ---------- (2) matched-actuator switch baseline, launched first so it runs all night
env $ACT DOUBLEBEE_REWARD_V2=1 DOUBLEBEE_NO_DR=1 DOUBLEBEE_EPISODE_S=12 DOUBLEBEE_W_E=4.0 \
    DOUBLEBEE_SWITCH_LOW=0.31 DOUBLEBEE_SWITCH_HIGH=1.0 DOUBLEBEE_SWITCH_THRESH=0.02 DOUBLEBEE_SWITCH_LATCH=3.0 \
    DOUBLEBEE_RESUME_PATH=$W40/model_3000.pt DOUBLEBEE_RUN_NAME=actfix_swA \
    nohup setsid $IL -p scripts/co_rl/train.py \
      --task Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-v1-ppo --algo tqc \
      --num_envs 1024 --max_iterations 2000 --headless --seed 42 \
      > sweep_logs/actfix/train_actfix_swA.log 2>&1 < /dev/null &
echo "[night] switch training launched $(date -u)" >> $LOG
sleep 60

# ---------- (1) evaluations, 3 at a time
ev () {  # run ckpt height tag
  OUT="actfix_eval/$4_c$2_h$3.csv"
  [ -s "$OUT" ] && return
  env $ACT timeout 3000 $IL -p scripts/paper/eval_climb.py \
     --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
     --checkpoint "$1/model_$2.pt" --step-height "0.$3" --episodes 200 --num_envs 8 \
     --seed 6000 --out "$OUT" > "actfix_eval/$4_c$2_h$3.log" 2>&1
}
n=0
for C in 2800 2900 3000 3100 3200 3300 3400 3500 3600; do ev $W40 $C 06 wE40 & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; wait
echo "[night] wE40 6cm pooled done $(date -u)" >> $LOG
for C in 3300 3400 3500 3600 3700; do for H in 05 07; do ev $W40 $C $H wE40 & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
echo "[night] wE40 5/7cm done $(date -u)" >> $LOG
CK=$(ls $W02/model_*.pt | sed "s/.*model_//;s/\.pt//" | sort -n | tail -10)
for C in $CK; do ev $W02 $C 06 wE025 & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; wait
echo "[night] wE025 6cm pooled done $(date -u)" >> $LOG

python3 - >> $LOG <<"PY"
import csv, glob, re, statistics as st
def cell(tag,h):
    rows=[]
    for f in sorted(glob.glob("actfix_eval/%s_c*_h%s.csv"%(tag,h))):
        r=list(csv.DictReader(open(f)))
        if not r: continue
        cl=100*sum(1 for x in r if float(x["max_gain_m"])>=int(h)/100)/len(r)
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r if float(x["energy_J"])>0]
        rows.append((int(re.search(r"_c(\d+)_",f).group(1)),cl,sum(pw)/len(pw) if pw else 0))
    if not rows: return None
    cls=[x[1] for x in rows]; pws=[x[2] for x in rows]
    se=st.pstdev(cls)/len(cls)**0.5 if len(cls)>1 else 0
    return len(rows),sum(cls)/len(cls),se,sum(pws)/len(pws),sorted(rows)
print("\n=== POOLED RESULTS (seed 6000, 200 eps/ckpt)")
for tag in ("wE40","wE025"):
    for h in ("05","06","07"):
        c=cell(tag,h)
        if c: print("%-6s %s cm  n=%2d ckpts  clears %.1f +- %.1f %%   power %.0f W"%(tag,h,c[0],c[1],c[2],c[3]))
        if c and h=="06": print("        per-ckpt:", ", ".join("%d:%.0f%%"%(k,v) for k,v,_ in c[4]))
PY
echo "[night] ALL DONE $(date -u)" >> $LOG
