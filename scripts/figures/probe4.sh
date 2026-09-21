cd /data/doubleBee/doubleBee_terr_spawn
echo "=== TRAINED RUNS matching prop/wheel-only"
find logs/co_rl -maxdepth 3 -type d | grep -i "prop\|wheelonly\|wheels_only\|wheelsonly\|flight" | head -20
echo "--- all run names (last 30)"
find logs/co_rl -maxdepth 3 -type d -name "2026-*" | sed 's|.*/||' | sort | tail -30
echo
echo "=== ENERGY SPLIT BY END REASON"
python3 - <<'PY'
import csv,glob,collections
def stats(f,label):
    r=list(csv.DictReader(open(f)))
    by=collections.defaultdict(list)
    for x in r: by[x["end"]].append(x)
    tot=len(r)
    gr=by.get("goal_reached",[])
    E=lambda rows: sum(float(x["energy_J"]) for x in rows)/len(rows) if rows else float("nan")
    T=lambda rows: sum(int(x["steps"]) for x in rows)/len(rows)*0.02 if rows else float("nan")
    P=lambda rows: sum(float(x["energy_J"])/max(1,int(x["steps"]))/0.02 for x in rows)/len(rows) if rows else float("nan")
    print("%-22s goal %3d/%d (%4.1f%%)  E_goal %6.0f J  t_goal %4.2f s  P_goal %4.0f W | E_all %6.0f J  P_all %4.0f W"%(
        label,len(gr),tot,100*len(gr)/tot,E(gr),T(gr),P(gr),E(r),P(r)))
for c in (3300,3400,3500,3600,3700): stats("actfix_eval/wE40_c%d_h06.csv"%c,"wE4 c%d"%c)
print()
for c in (3500,3600,3900,3999): stats("actfix_eval/wE025_c%d_h06.csv"%c,"wE0.25 c%d"%c)
print()
print("--- pooled energy per successful goal, and J per cm of useful climb (capped at 0.12 m)")
def pooled(pat,label):
    gE=gN=0; sE=sG=0
    for f in sorted(glob.glob(pat)):
        for x in csv.DictReader(open(f)):
            e=float(x["energy_J"]); g=min(float(x["max_gain_m"]),0.12)
            if x["end"]=="goal_reached": gE+=e; gN+=1
            sE+=e; sG+=g
    print("%-10s E/goal %6.0f J (n=%d)   pooled J per cm useful climb %5.0f"%(label,gE/max(1,gN),gN,sE/sG/100))
pooled("actfix_eval/wE40_c3[3-6]00_h06.csv","wE4 best")
pooled("actfix_eval/wE40_c*_h06.csv","wE4 all")
pooled("actfix_eval/wE025_c*_h06.csv","wE0.25")
PY
