cd /data/doubleBee/doubleBee_terr_spawn
echo "=== REGISTERED TASKS (HybridStair family)"
grep -rn "id=\"Isaac-Velocity-HybridStair" --include=*.py . 2>/dev/null | sed 's/:.*id="/  /;s/".*//' | sort -u
echo
echo "=== MODE / WHEEL-DISABLE ENV FLAGS"
grep -rn "DOUBLEBEE_[A-Z_]*" --include=*.py source scripts 2>/dev/null | grep -oE "DOUBLEBEE_[A-Z0-9_]+" | sort | uniq -c | sort -rn | head -40
echo
echo "=== prop-only / flight-only hints"
grep -rni "prop_only\|proponly\|flight_only\|no_wheel\|wheels_off\|disable_wheel" --include=*.py --include=*.sh . 2>/dev/null | head -20
echo
echo "=== eval_climb columns + thrust availability"
grep -n "fieldnames\|writerow\|thrust\|energy" scripts/paper/eval_climb.py | head -40
echo
echo "=== ROBOT MASS"
grep -rni "mass\b" --include=*.py source/*/*/*/*/doublebee* 2>/dev/null | head -10
grep -rn "9.81\|TOTAL_MASS\|WEIGHT_N" --include=*.py source scripts 2>/dev/null | grep -i "mass\|weight" | head -10
