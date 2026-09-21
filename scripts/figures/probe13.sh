cd /data/doubleBee/doubleBee_terr_spawn
echo "########## were the fixed-thrust arms warm-started?"
sed -n '1,45p' ct_matched_driver.sh
echo
echo "########## stage chain: what did each stage resume from"
grep -rn "RESUME_PATH\|WARM=" run_queue.sh scripts/paper/drivers/*.sh 2>/dev/null | head -20
echo
echo "########## deployed hardware checkpoint"
grep -rn -i "model_[0-9]*\.pt\|checkpoint" modified_mav_ros_src/*/*/db_inference.py 2>/dev/null | head -8
find . -name "*.pt" -path "*deploy*" -o -name "*.pt" -path "*hw*" 2>/dev/null | head
ls -la scripts/paper/hw/ 2>/dev/null | head
grep -rn -i "hE4\|model_5[0-9]*" scripts/paper/frozen_config.txt scripts/paper/RUNBOOK_switched_thrust.md 2>/dev/null | head
