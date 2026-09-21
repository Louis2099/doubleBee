cd /data/doubleBee/doubleBee_terr_spawn
python3 - <<'PY'
import csv, glob, math, statistics as st
H=0.06
print("FIG5 DATA (paired seeds, abl_seeded, 10 ckpts x 200 eps)\n")
print("LEFT: energy per episode by steps climbed (mean, SE across ckpts, n_eps)")
print("%-4s %-6s %10s %8s %7s"%("wE","group","E(J)","SE","n_eps"))
right={}
for tag,w in (("hE0","0"),("hE2","2"),("hE4","4"),("hE6","6"),("hE8","8")):
    buckets={0:[],1:[],2:[]}
    cnt={0:0,1:0,2:0}
    clears=[]; e1ck=[]
    for f in sorted(glob.glob("abl_seeded/climb_%s_h06_*.csv"%tag)):
        rr=list(csv.DictReader(open(f)))
        if not rr: continue
        per={0:[],1:[],2:[]}
        for x in rr:
            g=float(x["max_gain_m"]); e=float(x["energy_J"])
            k=0 if g<H else (1 if g<2*H else 2)
            per[k].append(e); cnt[k]+=1
        for k in (0,1,2):
            if per[k]: buckets[k].append(sum(per[k])/len(per[k]))
        clears.append(100*sum(1 for x in rr if float(x["max_gain_m"])>=H)/len(rr))
        ec=[float(x["energy_J"]) for x in rr if float(x["max_gain_m"])>=H]
        if ec: e1ck.append(sum(ec)/len(ec))
    for k,lab in ((0,"0"),(1,"1"),(2,"2+")):
        v=buckets[k]
        se=st.pstdev(v)/len(v)**0.5 if len(v)>1 else 0
        print("%-4s %-6s %10.1f %8.1f %7d"%(w,lab,sum(v)/len(v) if v else float("nan"),se,cnt[k]))
    cse=st.pstdev(clears)/len(clears)**0.5
    ese=st.pstdev(e1ck)/len(e1ck)**0.5
    right[w]=(sum(clears)/len(clears),cse,sum(e1ck)/len(e1ck),ese)
print("\nRIGHT: clearance vs energy on episodes clearing a step")
print("%-4s %10s %7s %10s %7s"%("wE","clears%","SE","E(J)","SE"))
for w in ("0","2","4","6","8"):
    c,cs,e,es=right[w]
    print("%-4s %10.1f %7.1f %10.1f %7.1f"%(w,c,cs,e,es))
PY
