import csv, math, statistics as st, sys

def load(p):
    with open(p) as f: return list(csv.DictReader(f))

def segs(rows, minlen=40):
    out=[]; cur=None
    for i,r in enumerate(rows):
        active = str(r['gated']).strip()=='0'
        if active:
            if cur is None: cur=[i,i]
            else: cur[1]=i
        elif cur is not None:
            out.append(tuple(cur)); cur=None
    if cur: out.append(tuple(cur))
    return [s for s in out if s[1]-s[0]>=minlen]

def tilt(r):
    qx,qy,qz,qw=(float(r[k]) for k in ('qx','qy','qz','qw'))
    zz=1-2*(qx*qx+qy*qy)
    return math.degrees(math.acos(max(-1,min(1,zz))))

def floor_z(rows):
    up=[float(r['pos_z']) for r in rows if tilt(r)<20]
    up.sort()
    return up[max(0,int(0.02*len(up)))] if up else min(float(r['pos_z']) for r in rows)

def report(path):
    rows=load(path); fz=floor_z(rows); S=segs(rows)
    print(f"--- {path.split('/')[-1]}  floor_z={fz:.3f}  segments={len(S)}")
    for k,(a,b) in enumerate(S):
        w=rows[a:b+1]
        z=[float(r['pos_z']) for r in w]; ti=[tilt(r) for r in w]
        # peak height reached while upright
        cand=[(z[i],i) for i in range(len(w)) if ti[i]<25]
        if not cand: print(f"  seg{k}: never upright"); continue
        zpk,ipk=max(cand)
        wc=w[:ipk+1]
        watts=[float(r['watts']) for r in wc if r['watts'] not in ('','nan')]
        P=st.mean(watts) if watts else float('nan')
        t0,t1=float(wc[0]['t']),float(wc[-1]['t'])
        travel=sum(math.hypot(float(wc[i+1]['pos_x'])-float(wc[i]['pos_x']),
                              float(wc[i+1]['pos_y'])-float(wc[i]['pos_y'])) for i in range(len(wc)-1))
        # descent after peak while still upright
        after=[(z[i],ti[i]) for i in range(ipk,len(w))]
        zend=z[-1]; drop=zpk-min(zz for zz,tt in after if tt<25) if any(tt<25 for _,tt in after) else 0
        print(f"  seg{k}: n={len(w)} t={t1-t0:5.2f}s  zpk={zpk:.3f} (above floor {zpk-fz:+.3f})"
              f"  P={P:5.0f}W E={P*(t1-t0):5.0f}J travel={travel:4.2f}m  drop_after={drop:.3f} zend={zend:.3f} tilt_end={ti[-1]:.0f}")

for p in sys.argv[1:]: report(p)
