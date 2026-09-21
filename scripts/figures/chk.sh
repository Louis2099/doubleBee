cd /data/doubleBee/doubleBee_terr_spawn
echo "=== live play.py: log_policy_io present? ==="
grep -n "log_policy_io\|policy_io_path\|header.append\|prop\|thrust\|joint_vel" scripts/co_rl/play.py | head -30
echo
echo "=== header construction in play.py ==="
sed -n "$(grep -n 'policy_io_header_written' scripts/co_rl/play.py | head -1 | cut -d: -f1),+30p" scripts/co_rl/play.py 2>/dev/null
echo
echo "=== optionC ==="
date -u; echo "phase2: $(ls abl_seeded/ | grep -c '^climb_ct050\|^climb_ctm05\|^climb_ctm45')/150"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
