import csv, glob, os
import numpy as np
C=[4.0792540792478203e-13,-2.2921522921483562e-09,2.2550699300690226e-05,
   -0.026882905982896884,8.516433566430207]
def p2t(p):
    p=np.clip(np.asarray(p,float),1000.,1650.); t=np.zeros_like(p)
    for c in C: t=t*p+c
    return np.maximum(t,0.)
W=31.57; M=3.2182; DT=0.02
def runs(mask):
    """longest continuous True run, in seconds"""
    if not mask.any(): return 0.0
    best=cur=0
    for v in mask:
        cur = cur+1 if v else 0
        best=max(best,cur)
    return best*DT

print("servo units check + vertical component of thrust\n")
print("%-18s %7s %7s | %6s %6s | %7s %7s | %7s %7s" % (
 "trial","|T|mean","|T|max","tiltmn","tiltmx","Tz mean","Tz max","|T|>W","Tz>W"))
rowsum=[]
for p in sorted(glob.glob("hw_final/trials_final/trial_*.csv")):
    r=list(csv.DictReader(open(p)))
    f=lambda k: np.array([float(x[k]) for x in r])
    live=np.flatnonzero(np.array([x["gated"]=="0" for x in r]))
    z=f("pos_z"); kpk=live[int(np.argmax(z[live]))]; seg=live[live<=kpk]
    if z[seg].max()-z[seg][:20].mean()<0.12: continue
    u=(f("u_thr1")[seg]+1)*325.+1000.; v=(f("u_thr2")[seg]+1)*325.+1000.
    T=p2t(u)+p2t(v)
    s1,s2=f("servo1")[seg],f("servo2")[seg]
    tilt=0.5*(np.abs(s1)+np.abs(s2))            # rad, servo limit 0.7854
    qx,qy,qz,qw=f("qx")[seg],f("qy")[seg],f("qz")[seg],f("qw")[seg]
    pitch=np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))
    Tz=T*np.cos(tilt+pitch)
    rowsum.append((T,Tz,tilt))
    print("%-18s %7.1f %7.1f | %6.2f %6.2f | %7.1f %7.1f | %6.1f%% %6.1f%%"%(
      os.path.basename(p),T.mean(),T.max(),tilt.min(),tilt.max(),
      Tz.mean(),Tz.max(),(T>W).mean()*100,(Tz>W).mean()*100))

T=np.concatenate([a for a,_,_ in rowsum]); Tz=np.concatenate([b for _,b,_ in rowsum])
tl=np.concatenate([c for _,_,c in rowsum])
print("\npooled over the 8 successes")
print("  |T|  mean %.1f N (T/W %.2f)   above weight %.0f%% of ticks"%(T.mean(),T.mean()/W,(T>W).mean()*100))
print("  Tz   mean %.1f N (T/W %.2f)   above weight %.0f%% of ticks"%(Tz.mean(),Tz.mean()/W,(Tz>W).mean()*100))
print("  tilt mean %.1f deg (max %.1f)"%(np.degrees(tl).mean(),np.degrees(tl).max()))

print("\nlongest continuous above-weight excursion, per trial (s)")
for p,(T_,Tz_,_) in zip(sorted(glob.glob("hw_final/trials_final/trial_*.csv"))[:0]+[x for x in sorted(glob.glob("hw_final/trials_final/trial_*.csv"))],rowsum) if False else []:
    pass
for (T_,Tz_,_),p in zip(rowsum,[f for f in sorted(glob.glob("hw_final/trials_final/trial_*.csv"))][:len(rowsum)]):
    print("  %-18s |T|>W %.2f s   Tz>W %.2f s"%(os.path.basename(p),runs(T_>W),runs(Tz_>W)))
