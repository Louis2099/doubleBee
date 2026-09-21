cd /data/doubleBee/doubleBee_terr_spawn
echo "=== scripts/paper"
ls -1 scripts/paper/
echo
echo "=== which runs/dirs each paper figure script reads"
for f in scripts/paper/fig_*.py scripts/paper/tab_*.py scripts/paper/table_*.py; do
  [ -f "$f" ] || continue
  echo "--- $f"
  grep -n -E "logs/co_rl|_eval|\.csv|RUNS|CKPT|abl_|hE4|sw[AB]|glob\.glob" "$f" | head -14
done
echo
echo "=== eval dirs present"
ls -d *_eval* 2>/dev/null
echo
echo "=== did the paper evals pass --seed?  (search driver scripts)"
grep -ln "eval_climb.py" *.sh sweep_logs/*/*.sh 2>/dev/null | head -20
echo "---"
for s in $(grep -ln "eval_climb.py" *.sh 2>/dev/null | head -12); do
  printf "%-28s seed=%s\n" "$s" "$(grep -o -- "--seed[= ][0-9]*" $s | head -1)"
done
echo
echo "=== eval_climb --seed default + what it controls"
grep -n -A4 "\-\-seed" scripts/paper/eval_climb.py | head -30
echo
echo "=== one eval run wall time"
for l in $(ls -t actfix_eval/*.log 2>/dev/null | head -3); do
  echo "$l  $(stat -c %y $l | cut -c1-19)"
done
