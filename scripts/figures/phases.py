import csv
import numpy as np
C=[4.0792540792478203e-13,-2.2921522921483562e-09,2.2550699300690226e-05,
   -0.026882905982896884,8.516433566430207]
def p2t(p):
    p=np.clip(np.asarray(p,float),1000.,1650.); t=np.zeros_like(p)
    for c in C: t=t*p+c
    return np.maximum(t,0.)
def sm(a,n=5): return np.convolve(a,np.ones(n)/n,mode="same")
DT=0.02; F1,F2=-0.55,-0.15; W=31.57
r=list(csv.DictReader(open("hw_final/trials_final/trial_190715.csv")))
f=lambda k: np.array([float(x[k]) for x in r])
live=np.flatnonzero(np.array([x["gated"]=="0" for x in r]))
seg=live                                   # FULL live segment, not just to peak
x,z=f("pos_x")[seg],f("pos_z")[seg]
qx,qy,qz,qw=f("qx")[seg],f("qy")[seg],f("qz")[seg],f("qw")[seg]
pitch=np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1)))
vx=sm(np.gradient(x,DT)); vz=sm(np.gradient(z,DT))
pr=np.radians(pitch)
vxb=vx*np.cos(pr)+vz*np.sin(pr); vzb=-vx*np.sin(pr)+vz*np.cos(pr)
T=p2t((f("u_thr1")[seg]+1)*325.+1000.)+p2t((f("u_thr2")[seg]+1)*325.+1000.)
t=np.arange(seg.size)*DT
z0=z[:20].mean(); kpk=int(np.argmax(z))
print("live %.2f s, peak at %.2f s, gain %.3f m, x %.2f -> %.2f"%(t[-1],t[kpk],z.max()-z0,x.min(),x.max()))
print("\ncoarse trace (every 0.2 s)")
print("%6s %7s %7s %7s %7s %7s %7s"%("t","x","z-z0","vx","vz","pitch","T"))
for i in range(0,len(t),10):
    print("%6.2f %7.3f %7.3f %7.2f %7.2f %7.1f %7.1f"%(t[i],x[i],z[i]-z0,vxb[i],vzb[i],pitch[i],T[i]))
print("\npost-peak: z end %.3f (peak %.3f), x end %.2f, pitch end %+.1f"%(
    z[-1]-z0,z.max()-z0,x[-1],pitch[-1]))
# phase boundaries from x
def first(mask,d=0):
    m=np.flatnonzero(mask); return m[0] if m.size else d
a_end=first(x>F1-0.10,0)
s1_end=first(x>F2-0.10,a_end)
s2_end=first((x>F2)&(z>z0+0.10),s1_end)
print("\nphases: approach 0-%.2f | step1 %.2f-%.2f | step2 %.2f-%.2f | settle %.2f-%.2f"%(
    t[a_end],t[a_end],t[s1_end],t[s1_end],t[s2_end],t[s2_end],t[-1]))
for nm,i0,i1 in (("approach",0,a_end),("step 1",a_end,s1_end),("step 2",s1_end,s2_end),("settle",s2_end,len(t))):
    if i1<=i0: print("  %-9s EMPTY"%nm); continue
    sl=slice(i0,i1)
    print("  %-9s %.2f-%.2f s | vx %+.2f..%+.2f | vz %+.2f..%+.2f | pitch %+.1f..%+.1f | T %.1f..%.1f mean %.1f | %.0f%% >W"%(
        nm,t[i0],t[i1-1],vxb[sl].min(),vxb[sl].max(),vzb[sl].min(),vzb[sl].max(),
        pitch[sl].min(),pitch[sl].max(),T[sl].min(),T[sl].max(),T[sl].mean(),(T[sl]>W).mean()*100))
