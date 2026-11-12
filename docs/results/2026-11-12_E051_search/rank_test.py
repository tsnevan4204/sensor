import sys, time, itertools, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '../../src/oned'))
import numpy as np, wallsolve as WS, treesolve as TS, fastbatch as FB, devtree as DT
C0=TS.C0
def run(seed,N=3,h=2.5e-3):
    f,y,s,tr=WS.make_problem(seed,N)
    fs,ys,ss,segs=TS.search_subset(f,y,s)
    lib=TS.Library(fs); k0=2*np.pi*fs/C0; w=1/ss**2
    ti=DT.truth_interfaces(tr)
    print(f"seed {seed}", [(L['name'],round(1e3*L['t'],1)) for L in tr['layers']], [round(1e3*g,1) for g in tr['gaps']], tr['back'])
    names=[c for _,c in ti]; pos=np.array([z for z,_ in ti]); d=tr['standoff']
    cidx=[lib.names.index(c) for c in names]
    nl=len(names)-1
    def pack(cl_list,pos_list,dd):
        P=len(cl_list); C=np.zeros((P,FB.LMAX),np.int64); T=np.zeros((P,FB.LMAX)); bk=np.zeros(P,np.int64)
        for p,(cl,ps) in enumerate(zip(cl_list,pos_list)):
            C[p,:nl]=cl[:nl]; bk[p]=cl[nl]; T[p,:nl]=np.diff(ps)/lib.nref[list(cl[:nl])]
        return C,T,np.full(P,nl),bk,np.full(P,dd)
    # truth at nominal classes, true optical positions
    c0=FB.chi_batch(*pack([cidx],[pos],d),lib.ntab,k0,ys,w)[0]
    n=2*fs.size
    print(f"  nominal-class misfit at true positions: chi2/n = {c0/n:.1f}")
    # (1) position grid for the true classes
    L=pos[-1]-pos[0]; grid=np.arange(h, L+0.06, h)
    combos=[c for c in itertools.combinations(range(len(grid)), nl) ]
    if len(combos)>3e6: combos=combos[::int(len(combos)/3e6)+1]
    PS=[np.concatenate([[d],d+grid[list(c)]]) for c in combos]
    t0=time.time(); chi=FB.chi_batch(*pack([cidx]*len(PS),PS,d),lib.ntab,k0,ys,w)
    o=np.argsort(chi)
    print(f"  positions grid ({len(PS)} combos, {time.time()-t0:.1f}s): true (mm) {np.round(1e3*(pos[1:]-d),1)}")
    for i in o[:4]: print(f"     chi2/n {chi[i]/n:8.1f}  {np.round(1e3*(PS[i][1:]-d),1)}")
    # (2) all class sequences at the true positions (same interface count)
    sol=list(lib.is_solid_idx); AIR,MET=lib.AIR,lib.METAL
    seqs=[]
    for cl in itertools.product(sol+[AIR], repeat=nl):
        if cl[0]==AIR or any(a==b for a,b in zip(cl[:-1],cl[1:])): continue
        if sum(1 for c in cl if c!=AIR)!=N: continue
        for bk in (AIR,MET):
            if bk==AIR and cl[-1]==AIR: continue
            seqs.append(tuple(cl)+(bk,))
    chi=FB.chi_batch(*pack(seqs,[pos]*len(seqs),d),lib.ntab,k0,ys,w); o=np.argsort(chi)
    rk=[k for k,i in enumerate(o) if seqs[i]==tuple(cidx)][0]
    print(f"  class sequences at true positions ({len(seqs)}): true rank {rk}")
    for i in o[:4]: print(f"     chi2/n {chi[i]/n:8.1f}  {[lib.names[c] for c in seqs[i]]}")
for sd in sys.argv[1:]: run(int(sd))
