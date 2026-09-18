#!/bin/bash
# Propeller ceiling probe: hold propellers at full command (ConstThrust, hold 1.0)
# and read achieved spin / PWM / thrust from the [action->thrust] diagnostic,
# which prints every 500 aerodynamics calls. Variants differ only in the new
# DOUBLEBEE_PROP_* / WHEEL_ARMATURE overrides. Runs all variants in parallel.
cd /data/doubleBee/doubleBee_terr_spawn
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
IL=../../isaaclab/IsaacLab/isaaclab.sh
CK=$(ls -dt logs/co_rl/doublebee_velocity/tqc/*_abl_ct10 | head -1)/model_3999.pt
mkdir -p /tmp/probe
run () {  # tag envstring
  ( env DOUBLEBEE_HOLD_ACTION=1.0 $2 timeout 900 $IL -p scripts/paper/eval_climb.py \
      --task Isaac-Velocity-HybridStair-DoubleBee-ConstThrust-Play-v1-ppo \
      --checkpoint "$CK" --step-height 0.03 --episodes 40 --num_envs 8 \
      --out /tmp/probe/$1.csv > /tmp/probe/$1.log 2>&1 ) &
}
run base      "DOUBLEBEE_NOOP=1"
run e25d16a16 "DOUBLEBEE_PROP_EFFORT=25 DOUBLEBEE_PROP_DAMPING=0.16 DOUBLEBEE_PROP_ARMATURE=0.0016"
run e30d20a20 "DOUBLEBEE_PROP_EFFORT=30 DOUBLEBEE_PROP_DAMPING=0.20 DOUBLEBEE_PROP_ARMATURE=0.0020"
run e25d16a05 "DOUBLEBEE_PROP_EFFORT=25 DOUBLEBEE_PROP_DAMPING=0.16 DOUBLEBEE_PROP_ARMATURE=0.0005"
wait
python3 - <<'PY'
import re, glob, statistics as st
pat=re.compile(r"joint_vel\(rad/s\)=\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\].*?PWM=\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\].*?thrust\(N\)=\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s*\]")
for f in sorted(glob.glob("/tmp/probe/*.log")):
    t=open(f,errors="ignore").read()
    rows=[tuple(abs(float(x)) for x in m.groups()) for m in pat.finditer(t)]
    tb="TRACEBACK" if "Traceback" in t else ""
    if not rows:
        print("%-8s no samples %s" % (f.split('/')[-1], tb)); continue
    w=[(r[0]+r[1])/2 for r in rows]; p=[(r[2]+r[3])/2 for r in rows]; th=[r[4]+r[5] for r in rows]
    sd=st.pstdev(w) if len(w)>1 else 0.0
    print("%-10s n=%2d  spin median %5.0f sd %4.0f min %5.0f max %5.0f rad/s | PWM median %4.0f | total thrust median %5.1f N (T/W %.2f) %s" % (
        f.split('/')[-1].replace('.log',''), len(rows), st.median(w), sd, min(w), max(w), st.median(p), st.median(th), st.median(th)/31.57, tb))
PY
echo "[probe] DONE $(date -u)"
