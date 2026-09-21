import csv, math, sys, statistics as st

def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))

def segs(rows):
    out=[]; cur=None
    for i,r in enumerate(rows):
        g=str(r.get('gated','')).strip().lower() in ('1','true','t','yes')
        if g:
            if cur is None: cur=[i,i]
            else: cur[1]=i
        else:
            if cur is not None: out.append(tuple(cur)); cur=None
    if cur is not None: out.append(tuple(cur))
    return [s for s in out if s[1]-s[0] > 20]

def tilt(r):
    qx,qy,qz,qw=(float(r[k]) for k in ('qx','qy','qz','qw'))
    # angle between body z and world z
    zz = 1-2*(qx*qx+qy*qy)
    return math.degrees(math.acos(max(-1,min(1,zz))))

def score(path, seg_idx):
    rows=load(path); S=segs(rows)
    if seg_idx=='last': seg_idx=len(S)-1
    a,b=S[seg_idx]
    win=rows[a:b+1]
    zs=[float(r['pos_z']) for r in win]
    pk=max(range(len(zs)), key=lambda i: zs[i])
    w=win[:pk+1]
    watts=[float(r['watts']) for r in w if r['watts'] not in ('','nan')]
    t0,t1=float(w[0]['t']),float(w[-1]['t'])
    dx=float(w[-1]['pos_x'])-float(w[0]['pos_x']); dy=float(w[-1]['pos_y'])-float(w[0]['pos_y'])
    travel=sum(math.hypot(float(w[i+1]['pos_x'])-float(w[i]['pos_x']),
                          float(w[i+1]['pos_y'])-float(w[i]['pos_y'])) for i in range(len(w)-1))
    P=st.mean(watts) if watts else float('nan')
    return dict(nseg=len(S), P=P, dur=t1-t0, E=P*(t1-t0), travel=travel,
                zpk=zs[pk], tilt_pk=tilt(w[-1]))

groups={
 'narrow678': [('trials678/trial_234446.csv','last'),('trials678/trial_000701.csv','last'),
               ('trials678/trial_001803.csv','last'),('trials678/trial_002434.csv','last'),
               ('trials678/trial_002232.csv','last')],
 'ramp':      [('ramp/trial_202042.csv',2),('ramp/trial_203011.csv',1),('ramp/trial_202312.csv',0),
               ('ramp/trial_200520.csv',0),('ramp/trial_202554.csv',0)],
 'foam89':    [('trials89/trial_211902.csv',0),('trials89/trial_204512.csv',1),
               ('trials89/trial_211112.csv',0),('trials89/trial_210158.csv',0),
               ('trials89/trial_205801.csv',4)],
}
base='/home/airlab/doublebee_PID_JAI/hw_final/trials_final/'
for g,items in groups.items():
    print('==',g)
    Ps=[];Es=[];Ts=[];Tr=[]
    for p,si in items:
        try:
            r=score(base+p,si)
        except Exception as e:
            print('  ',p,'ERR',e); continue
        print('   %-34s nseg=%d P=%5.0f W  t=%4.2f s  E=%5.0f J  travel=%4.2f m  zpk=%.3f tilt=%.0f'
              %(p.split('/')[-1],r['nseg'],r['P'],r['dur'],r['E'],r['travel'],r['zpk'],r['tilt_pk']))
        Ps.append(r['P']);Es.append(r['E']);Ts.append(r['dur']);Tr.append(r['travel'])
    if Ps:
        f=lambda v:(st.mean(v), st.stdev(v) if len(v)>1 else 0)
        print('   MEAN P %.0f +- %.0f W | t %.2f +- %.2f s | E %.0f +- %.0f J | travel %.2f +- %.2f m'
              %(*f(Ps),*f(Ts),*f(Es),*f(Tr)))
