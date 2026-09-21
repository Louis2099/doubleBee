cd /data/doubleBee/doubleBee_terr_spawn
python3 - <<'PY'
import csv, glob, statistics as st
H=0.06
NAMES={"hE4":"Learned wE=4","ct10":"Fixed 0.55","ct050":"Fixed 0.46",
       "ctm05":"Fixed 0.31","ctm45":"Fixed 0.23","swA3":"Switch 0.31","swB3":"Switch 0.46"}
print("CLEARANCE, three definitions (6 cm, paired seeds)\n")
print("%-14s %10s %10s %10s"%("arm","peak>=6cm","held(0.6x)","cleared flag"))
for tag in ("hE4","swA3","swB3","ct10","ct050","ctm05","ctm45"):
    a=[];b=[];c=[]
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv"%tag)):
        rr=list(csv.DictReader(open(f)))
        if not rr: continue
        n=len(rr)
        a.append(100*sum(1 for x in rr if float(x["max_gain_m"])>=H)/n)
        b.append(100*sum(1 for x in rr if float(x["max_gain_m"])>=H
                 and float(x["end_gain_m"])>=0.6*float(x["max_gain_m"]))/n)
        c.append(100*sum(1 for x in rr if int(x["cleared"])==1)/n)
    if not a: continue
    print("%-14s %10.1f %10.1f %10.1f"%(NAMES[tag],
        sum(a)/len(a),sum(b)/len(b),sum(c)/len(c)))

print("\n\nFIG5 GROUPS: raw peak vs held (6 cm), episode counts and energy\n")
print("%-4s %-6s %8s %8s %8s %8s"%("wE","grp","n_raw","E_raw","n_held","E_held"))
for tag,w in (("hE0","0"),("hE2","2"),("hE4","4"),("hE6","6"),("hE8","8")):
    raw={0:[],1:[],2:[]}; hld={0:[],1:[],2:[]}
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv"%tag)):
        for x in csv.DictReader(open(f)):
            g=float(x["max_gain_m"]); e=float(x["energy_J"]); eg=float(x["end_gain_m"])
            k=0 if g<H else (1 if g<2*H else 2)
            raw[k].append(e)
            gh=g if eg>=0.6*g else 0.0
            kh=0 if gh<H else (1 if gh<2*H else 2)
            hld[kh].append(e)
    for k,lab in ((0,"0"),(1,"1"),(2,"2+")):
        print("%-4s %-6s %8d %8.0f %8d %8.0f"%(w,lab,len(raw[k]),
            sum(raw[k])/len(raw[k]) if raw[k] else float("nan"),
            len(hld[k]), sum(hld[k])/len(hld[k]) if hld[k] else float("nan")))
PY
