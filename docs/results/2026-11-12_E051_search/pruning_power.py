"""Pruning-power experiment for a gated branch-and-bound: how many partial walls (prefixes of k
interfaces + half-space) survive the Slepian-gate lower bound at threshold = c * truth nominal cost."""
import sys, time, itertools, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '../../src/oned'))
import numpy as np, wallsolve as WS, treesolve as TS, fastbatch as FB, devtree as DT, wall_inverse as I
C0=TS.C0
def run(seed,N=3,h=5e-3,eps=1e-4,cthr=1.5,maxdepth=4):
    f,y,s,tr=WS.make_problem(seed,N); fs,ys,ss,segs=TS.search_subset(f,y,s)
    lib=TS.Library(fs); k0=2*np.pi*fs/C0; w=1/ss**2; d=tr['standoff']+0.0005
    ti=DT.truth_interfaces(tr); names=[c for _,c in ti]; pos=np.array([z for z,_ in ti])
    cidx=[lib.names.index(c) for c in names]; nl=len(names)-1
    C=np.array([cidx[:nl]]); T=np.array([np.diff(pos)/lib.nref[cidx[:nl]]])
    c0=FB.chi_batch(C,T,np.array([nl]),np.array([cidx[nl]]),np.array([tr['standoff']]),lib.ntab,k0,ys,w)[0]
    thr=cthr*c0
    ext=max(pos[-1]-pos[0]+0.03, 0.06); m=int(ext/h)
    print(f"seed {seed} {[(L['name'],round(1e3*L['t'],1)) for L in tr['layers']]} gaps {[round(1e3*g,1) for g in tr['gaps']]} {tr['back']}; truth nominal chi2 {c0:.0f} (chi2/n {c0/(2*fs.size):.1f}), thr {thr:.0f}, m={m}")
    a=d-0.05
    Qs=[]
    for q in range(m+2):
        rows=[]
        for (i0,i1) in segs:
            Qb,_=TS.slepian(fs[i0:i1],a,d+h*(q+1),eps); full=np.zeros((Qb.shape[0],fs.size),complex); full[:,i0:i1]=Qb/ss[i0:i1][None,:]; rows.append(full)
        Qs.append(np.ascontiguousarray(np.vstack(rows)))
    print("  gate dims K at q=0,4,8,..:", [Qs[q].shape[0] for q in range(0,m+2,4)])
    sol=list(lib.is_solid_idx); AIR,MET=lib.AIR,lib.METAL
    # nodes: (classes tuple incl. half-space class, positions q tuple (grid idx of interfaces after the front), nsolid)
    nodes=[((c,),(),1) for c in sol]
    for depth in range(1,maxdepth+1):
        # children: add interface at q>last, class behind
        byq={}
        for cls,qs,ns in nodes:
            last=qs[-1] if qs else -1
            cur=cls[-1]
            if cur==MET: continue
            opts=[]
            if cur!=AIR: opts.append((AIR,ns))
            if ns<N: opts+=[(c,ns+1) for c in sol if c!=cur]
            if ns==N: opts.append((MET,ns))
            for q in range(last+1,m):
                for c,ns2 in opts:
                    byq.setdefault(q,[]).append((cls+(c,),qs+(q,),ns2))
        surv=[]; tot=0
        for q,lst in byq.items():
            P=len(lst); tot+=P
            L=len(lst[0][0])-1
            Cc=np.array([n[0][:L] for n in lst],np.int64)
            tau=np.array([[0]+[h*(x+1) for x in n[1]] for n in lst])
            Tt=np.ascontiguousarray(np.diff(tau,axis=1)/lib.nref[Cc])
            bk=np.array([n[0][L] for n in lst],np.int64)
            Q=Qs[q+1]
            lb=FB.proj_batch(Cc,Tt,np.full(P,L),bk,np.full(P,d),lib.ntab,k0,Q@ys,Q)
            keep=np.where(lb<=thr)[0]
            surv+= [lst[i] for i in keep]
        # is the truth prefix among survivors?
        tq=[int(round((p-pos[0])/h))-1 for p in pos[1:depth+1]]
        tc=tuple(cidx[:depth+1])
        hit=any(n[0]==tc and all(abs(a_-b_)<=1 for a_,b_ in zip(n[1],tq)) for n in surv) if depth<=nl else None
        print(f"  depth {depth}: children {tot:.2e}, survive {len(surv):.2e}, truth prefix alive: {hit}", flush=True)
        nodes=surv
        if len(nodes)>3e5: print("  (stopping: too many)"); break
for sd in sys.argv[1:]: run(int(sd))
