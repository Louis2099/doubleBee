cd /data/doubleBee/doubleBee_terr_spawn
echo "########## optionC progress"
grep -c "^wrote" sweep_logs/switch/optionC.log 2>/dev/null
tail -3 sweep_logs/switch/optionC.log
ls abl_seeded/climb_hE[0268]_h06_*.csv 2>/dev/null | wc -l
echo
echo "########## hardware trial logs"
ls -d hw_final hw_logs scripts/paper/hw 2>/dev/null
find . -maxdepth 3 -path ./logs -prune -o \( -name "hw_*.csv" -o -name "trial*.csv" -o -name "*trial*.csv" \) -print 2>/dev/null | head -30
echo
echo "########## hw_trials.py expectations"
head -40 scripts/paper/hw_trials.py
