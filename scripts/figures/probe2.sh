cd /data/doubleBee/doubleBee_terr_spawn
echo "=== CKPT DIRS (depth 3)"
find logs/co_rl -maxdepth 3 -type d \( -name "*actfix*" \) | while read d; do
  n=$(ls "$d"/model_*.pt 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "$d : $n ckpts, last=$(ls "$d"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)"
done
echo
echo "=== NIGHT LOG"
cat sweep_logs/actfix/night.log
echo
echo "=== last lines of train logs"
for t in actfix_swA actfix_fixed055; do echo "--- $t"; grep -a -E "Learning iteration|Saving|Traceback|Error" sweep_logs/actfix/train_$t.log | tail -4; done
echo
echo "=== END REASONS + GAIN PROFILE"
python3 - <<'PY'
import csv,glob,collections
def show(f):
    r=list(csv.DictReader(open(f)))
    g=[float(x["max_gain_m"]) for x in r]
    eg=[float(x["end_gain_m"]) for x in r]
    dx=[float(x["max_disp_m"]) for x in r]
    e=collections.Counter(x["end"] for x in r)
    fly=100*sum(1 for v in g if v>=0.30)/len(r)
    print("%-24s clears %5.1f%%  fly>30cm %5.1f%%  maxgain %.3f  endgain %.3f  disp %.2f  ends %s"%(
      f.split("/")[1],100*sum(1 for v in g if v>=0.06)/len(r),fly,
      sum(g)/len(g),sum(eg)/len(eg),sum(dx)/len(dx),dict(e.most_common(4))))
for f in sorted(glob.glob("actfix_eval/wE025_c*_h06.csv")): show(f)
print()
for c in (3300,3400,3500,3600,3700): show("actfix_eval/wE40_c%d_h06.csv"%c)
PY
