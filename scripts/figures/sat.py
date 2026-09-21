import csv, glob, os
import numpy as np
C=[4.0792540792478203e-13,-2.2921522921483562e-09,2.2550699300690226e-05,
   -0.026882905982896884,8.516433566430207]
def p2t(p):
    p=np.clip(np.asarray(p,float),1000.,1650.); t=np.zeros_like(p)
    for c in C: t=t*p+c
    return np.maximum(t,0.)
FLOOR=p2t(1188.)*2; CEIL=p2t(1650.)*2
print("floor %.1f N   ceiling %.1f N\n"%(FLOOR,CEIL))
print("%-20s %7s %7s %7s | %-34s" % ("trial","%floor","%ceil","%mid","raw policy propeller action"))
agg=[]
for p in sorted(glob.glob("hw_final/trials_final/trial_*.csv")):
    r=list(csv.DictReader(open(p)))
    f=lambda k: np.array([float(x[k]) for x in r])
    live=np.flatnonzero(np.array([x["gated"]=="0" for x in r]))
    z=f("pos_z"); kpk=live[int(np.argmax(z[live]))]; seg=live[live<=kpk]
    if z[seg].max()-z[seg][:20].mean() < 0.12: continue
    u=(f("u_thr1")[seg]+1)*325.+1000.; v=(f("u_thr2")[seg]+1)*325.+1000.
    th=p2t(u)+p2t(v)
    lo=(th<=FLOOR+0.3).mean()*100; hi=(th>=CEIL-0.3).mean()*100
    mid=100-lo-hi
    acts={k:f(k)[seg] for k in ("action_0","action_1","action_2","action_3")}
    # propeller actions are the ones whose deployed image is u_thr; identify by correlation
    best=max(acts, key=lambda k: abs(np.corrcoef(acts[k],u)[0,1]) if acts[k].std()>1e-9 else 0)
    a=acts[best]
    a_sat=((a>=0.0).mean()*100)
    print("%-20s %6.1f%% %6.1f%% %6.1f%% | %s range [%+.2f,%+.2f] sd %.2f, %.0f%% of ticks a>=0"%(
        os.path.basename(p),lo,hi,mid,best,a.min(),a.max(),a.std(),a_sat))
    agg.append((lo,hi,mid,a_sat,a.std()))
A=np.array(agg)
print("\nmean over the 8 successes: floor %.0f%%  ceiling %.0f%%  modulated %.0f%%"%(A[:,0].mean(),A[:,1].mean(),A[:,2].mean()))
print("policy action sd %.2f, and %.0f%% of ticks command a>=0 (which the x2 scale maps to full thrust)"%(A[:,4].mean(),A[:,3].mean()))
