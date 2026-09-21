import csv, glob, os
import numpy as np
C=[4.0792540792478203e-13,-2.2921522921483562e-09,2.2550699300690226e-05,
   -0.026882905982896884,8.516433566430207]
def p2t(p):
    p=np.clip(np.asarray(p,float),1000.,1650.); t=np.zeros_like(p)
    for c in C: t=t*p+c
    return np.maximum(t,0.)
def sm(a,n=9):
    return np.convolve(a,np.ones(n)/n,mode="same")
W=31.57; M=3.2182; DT=0.02; S1X,S2X=-0.55,-0.15
print("Is the commanded excess thrust actually lifting the robot?\n")
print("%-18s %8s %8s %8s | %10s %10s" % (
 "trial","az|burst","az|rest","predicted","burst in band","burst total"))
allb=[];allr=[];inb=[];tot=[]
for p in sorted(glob.glob("hw_final/trials_final/trial_*.csv")):
    r=list(csv.DictReader(open(p)))
    f=lambda k: np.array([float(x[k]) for x in r])
    live=np.flatnonzero(np.array([x["gated"]=="0" for x in r]))
    z=f("pos_z"); kpk=live[int(np.argmax(z[live]))]; seg=live[live<=kpk]
    if z[seg].max()-z[seg][:20].mean()<0.12: continue
    zz=z[seg]; x=f("pos_x")[seg]
    u=(f("u_thr1")[seg]+1)*325.+1000.; v=(f("u_thr2")[seg]+1)*325.+1000.
    T=p2t(u)+p2t(v)
    s1,s2=f("servo1")[seg],f("servo2")[seg]
    tilt=0.5*(np.abs(s1)+np.abs(s2))
    qx,qy,qz,qw=f("qx")[seg],f("qy")[seg],f("qz")[seg],f("qw")[seg]
    pitch=np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))
    Tz=T*np.cos(tilt+pitch)
    az=sm(np.gradient(sm(np.gradient(zz,DT)),DT))
    b=Tz>W
    pred=(Tz[b]-W).mean()/M if b.any() else np.nan
    # is the burst near a step face?
    near=(np.abs(x-S1X)<0.18)|(np.abs(x-S2X)<0.18)
    fb=(b&near).sum()/max(b.sum(),1)*100
    allb.append(az[b]); allr.append(az[~b]); inb.append(fb); tot.append(b.sum()*DT)
    print("%-18s %8.2f %8.2f %8.2f | %9.0f%% %9.2f s"%(
      os.path.basename(p),az[b].mean(),az[~b].mean(),pred,fb,b.sum()*DT))
A=np.concatenate(allb); R=np.concatenate(allr)
print("\npooled (m/s^2): az during above-weight ticks %.2f,  az otherwise %.2f"%(A.mean(),R.mean()))
print("               predicted if thrust fully realised and wheels free: ~1.6")
print("bursts occurring within 18 cm of a step face: %.0f%% (mean over trials)"%np.mean(inb))
print("time above weight per trial: %.2f s mean, of windows 2.9-8.4 s"%np.mean(tot))
