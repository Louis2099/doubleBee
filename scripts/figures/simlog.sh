cd /data/doubleBee/doubleBee_terr_spawn
echo "=== who writes policy_io / pose logs ==="
grep -rn "policy_io\|POSE_LOG\|pose_log" --include=*.py scripts lab 2>/dev/null | head -20
echo
echo "=== existing policy_io files ==="
find . -maxdepth 3 -name "policy_io*" -o -maxdepth 3 -name "*pose_log*" 2>/dev/null | head -10
echo
echo "=== fig_climb_profile.py expected columns ==="
sed -n '1,60p' scripts/paper/fig_climb_profile.py
