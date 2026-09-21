cd /data/doubleBee/doubleBee_terr_spawn
B=logs/co_rl/doublebee_velocity/tqc
echo "########## checkpoint ranges per run"
for d in $B/energy_abl/*_hE0 $B/energy_abl/*_hE2 $B/energy_abl/*_hE4 $B/energy_abl/*_hE6 $B/energy_abl/*_hE8 \
         $B/energy_abl/*_gE4 $B/energy_abl/*_wE0 \
         $B/*_abl_ct10 $B/*_abl_ct050 $B/*_abl_ctm05 $B/*_abl_ctm45 \
         $B/*_abl_swA3 $B/*_abl_swB3 $B/*_abl_po $B/*_abl_wo $B/*_abl_ws; do
  [ -d "$d" ] || continue
  cks=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n)
  [ -z "$cks" ] && continue
  printf "%-46s %5s .. %-5s  (n=%d)\n" "$(basename $d)" "$(echo "$cks"|head -1)" "$(echo "$cks"|tail -1)" "$(echo "$cks"|wc -l)"
done
echo
echo "########## saved config / resume path per run"
for d in $B/energy_abl/*_hE4 $B/*_abl_ct10 $B/*_abl_swA3; do
  echo "--- $(basename $d)"; ls -1 "$d" | grep -v "^model_\|tfevents" | head -8
  for c in "$d"/params/env.yaml "$d"/params/agent.yaml "$d"/config.json "$d"/args.txt; do
    [ -f "$c" ] && echo "  found $c"
  done
done
echo
echo "########## launch commands / resume paths from driver scripts"
grep -rn "RESUME_PATH\|max_iterations\|RUN_NAME" *.sh scripts/paper/drivers/*.sh 2>/dev/null | grep -i "hE\|abl_ct\|abl_sw\|energy_abl" | head -25
echo
echo "########## which checkpoint was deployed to hardware"
grep -rn -i "checkpoint\|model_\|\.pt\b" scripts/paper/frozen_config.txt 2>/dev/null | head -10
ls scripts/paper/hw 2>/dev/null | head
grep -rn -i "model_.*\.pt\|policy.*\.onnx\|\.jit" modified_mav_ros_src 2>/dev/null | grep -o "model_[0-9]*\.pt\|[a-z_]*\.jit\|[a-z_]*\.onnx" | sort -u | head
