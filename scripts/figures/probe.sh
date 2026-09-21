cd /data/doubleBee/doubleBee_terr_spawn
echo "=== RUNNING"
pgrep -af "train.py" | cut -c1-140
pgrep -af "eval_climb.py" | cut -c1-160
pgrep -af actfix_night | cut -c1-100
echo
echo "=== CKPTS"
for d in $(find logs/co_rl -maxdepth 1 -type d -name "*actfix_swA*" -o -maxdepth 1 -type d -name "*actfix_fixed055*"); do
  echo "$d : $(ls $d/model_*.pt 2>/dev/null | wc -l) ckpts, last=$(ls $d/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)"
done
echo
echo "=== EVAL CSVS"
ls -1 actfix_eval/*.csv 2>/dev/null | sed 's|actfix_eval/||' | tr '\n' ' '; echo
echo
echo "=== wE025 DETAIL"
python3 - <<'PY'
import csv,glob,collections
for pat,h in (("wE025_c*_h06.csv",0.06),("wE40_c3500_h06.csv",0.06),("wE40_c3300_h06.csv",0.06)):
    for f in sorted(glob.glob("actfix_eval/"+pat)):
        r=list(csv.DictReader(open(f)))
        if not r: continue
        g=[float(x["max_gain_m"]) for x in r]
        pw=[float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in r]
        en=[float(x["energy_J"]) for x in r]
        e=collections.Counter(x.get("termination","?") for x in r)
        print("%-26s n=%3d clears %5.1f%% gain %.3f/%.3f pow %4.0fW E %4.0fJ ends %s"%(
          f.split("/")[1],len(r),100*sum(1 for v in g if v>=h)/len(r),
          sum(g)/len(g),max(g),sum(pw)/len(pw),sum(en)/len(en),dict(e.most_common(3))))
PY
echo
echo "=== CSV HEADER"
head -1 actfix_eval/wE025_c3500_h06.csv 2>/dev/null
