import csv, glob, os
import numpy as np
for E2 in (0.1312, 0.4312):
    print("\n--- step2 at x=%.4f ---" % E2)
    print("%-30s %6s %7s %7s %7s %6s %7s %7s %s" % (
        "file","live","dur_s","dist_m","E_J","P_W","z_max","x_max","both"))
    got=[]
    for p in sorted(glob.glob("hw_final/trials_final/*.csv"), key=os.path.getmtime):
        rows=list(csv.DictReader(open(p)))
        if len(rows)<40: continue
        col=rows[0].keys()
        f=lambda k: np.array([float(r[k]) for r in rows])
        live=np.array([r.get("gate_reason","")=="" for r in rows])
        s=np.flatnonzero(live)
        if s.size<20: continue
        x,z=f("pos_x"),f("pos_z")
        w=f("watts") if "watts" in col else np.zeros(len(rows))
        e=f("energy_J") if "energy_J" in col else np.zeros(len(rows))
        b=((x[s]>E2)&(z[s]>0.15)).sum()*0.02
        pw=w[s][w[s]>0].mean() if (w[s]>0).any() else float("nan")
        ej=e[s].max()-e[s].min()
        print("%-30s %6d %7.2f %7.3f %7.0f %6.0f %7.3f %7.3f %s"%(
            os.path.basename(p),s.size,s.size*0.02,x[s].max()-x[s].min(),ej,pw,
            z[s].max(),x[s].max(),"YES %.2fs"%b if b>0.2 else "-"))
        got.append((b>0.2,s.size*0.02,ej,pw,x[s].max()-x[s].min()))
    if got:
        ok=[g for g in got if g[0]]
        print("  cleared both: %d/%d"%(len(ok),len(got)))
        if ok:
            d=np.array([g[1] for g in ok]); e=np.array([g[2] for g in ok])
            p_=np.array([g[3] for g in ok]); ds=np.array([g[4] for g in ok])
            sd=lambda a: a.std(ddof=1) if len(a)>1 else 0.0
            print("  dur %.2f+-%.2f s  E %.0f+-%.0f J  P %.0f+-%.0f W  dist %.2f-%.2f m"%(
                d.mean(),sd(d),e.mean(),sd(e),p_.mean(),sd(p_),ds.min(),ds.max()))
