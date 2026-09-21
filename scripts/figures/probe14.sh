cd /data/doubleBee/doubleBee_terr_spawn
B=logs/co_rl/doublebee_velocity/tqc
echo "########## RUNBOOK stage chain"
sed -n '200,230p;305,345p' scripts/paper/RUNBOOK_switched_thrust.md
echo
echo "########## did the MATCHED fixed arms (ctA*) ever get trained or evaluated?"
ls -d $B/*abl_ctA* 2>/dev/null || echo "  NO ctA* run dirs"
ls abl_h/*ctA* 2>/dev/null || echo "  NO ctA* eval CSVs"
tail -6 sweep_logs/switch/ct_matched.log 2>/dev/null || echo "  no ct_matched.log"
echo
echo "########## recipe diff: hE4 vs abl_ct10 vs abl_swA3"
for d in $B/energy_abl/2026-09-06_23-10-57_hE4 $B/2026-09-08_07-10-49_abl_ct10 $B/2026-09-13_06-00-57_abl_swA3; do
  f=$d/params/env.yaml
  echo "--- $(basename $d)"
  grep -E "^\s*(episode_length_s|decimation|dt):" $f | head -4
  python3 - "$f" <<'PY'
import sys,re
t=open(sys.argv[1]).read()
for k in ("episode_length_s",):
    m=re.search(r"%s:\s*([0-9.]+)"%k,t)
    print("   %s = %s"%(k,m.group(1) if m else "?"))
# count active events (randomisation) and note reward term count
print("   events blocks:", len(re.findall(r"^\s{2}\w+:\s*$",t,re.M)))
for k in ("push_robot","add_base_mass","randomize_rigid_body_material","base_external_force_torque"):
    print("   %-32s %s"%(k,"present" if k in t else "ABSENT"))
m=re.search(r"energy\w*:\s*\n(?:.*\n)*?\s*weight:\s*(-?[0-9.]+)",t)
print("   energy weight:", m.group(1) if m else "?")
PY
done
