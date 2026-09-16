
"""make_data.py -- regenerates reference_solution.npz from first principles.
Every constant is a literal here; the cycle schedule is built analytically.
Scheme: explicit FTCS in the rock + ghost-node convective surface BC,
Dirichlet Tinf at x=L, forward Euler for the lumped air energy balance,
nsub=2 sub-steps per stored step (identical to sens_worker.py).
"""
import numpy as np

R, CP, CV      = 287.0, 1005.0, 718.0
V, AW          = 3.1e5, 4.0e4
P0, T0, TINF   = 4.6e6, 313.0, 313.0
TIN            = 323.0
K_ROCK, ALPHA  = 2.2, 3.0e-6
HW_TRUE        = 30.0
TC, NCYC, DT   = 86400.0, 20, 60.0
NR             = 120
MDOT_IN        = 108.0
MDOT_OUT       = 432.0
T_CHARGE       = 8*3600.0
T_DIS_START    = 16*3600.0
T_DISCHARGE    = 2*3600.0
L              = 15.0*np.sqrt(ALPHA*TC)
M0             = P0*V/(R*T0)

def schedule(t):
    """8 h charge [0,8) | idle [8,16) | 2 h discharge [16,18) | idle [18,24)"""
    tt = np.asarray(t) % TC
    mi = np.where(tt < T_CHARGE, MDOT_IN, 0.0)
    mo = np.where((tt >= T_DIS_START) & (tt < T_DIS_START+T_DISCHARGE), MDOT_OUT, 0.0)
    return mi, mo

def solve(hw=HW_TRUE, nsub=2, nr=NR, dt=DT, ncyc=NCYC):
    xr = np.linspace(0.0, L, nr); dx = xr[1]-xr[0]
    nt = int(ncyc*TC/dt)+1
    tv = np.arange(nt)*dt
    Min, Mout = schedule(tv)
    h  = dt/nsub
    c1 = ALPHA*h/dx**2
    c2 = 2.0*ALPHA*h*hw/(K_ROCK*dx)
    Tr = np.full(nr, TINF); T = T0; m = M0
    TrS = np.empty((nr, nt)); TS = np.empty(nt); mS = np.empty(nt); qS = np.empty(nt)
    TrS[:,0]=Tr; TS[0]=T; mS[0]=m; qS[0]=hw*(T-Tr[0])
    for i in range(nt-1):
        mi, mo = Min[i], Mout[i]
        for _ in range(nsub):
            Ts0 = Tr[0]
            dT  = (mi*(CP*TIN - CV*T) - mo*R*T - hw*AW*(T-Ts0))/(m*CV)
            Tn  = Tr.copy()
            Tn[1:-1] = Tr[1:-1] + c1*(Tr[2:] - 2*Tr[1:-1] + Tr[:-2])
            Tn[0]    = Tr[0] + 2*c1*(Tr[1]-Tr[0]) + c2*(T-Tr[0])
            Tn[-1]   = TINF
            T = T + h*dT; m = m + h*(mi-mo); Tr = Tn
        TrS[:,i+1]=Tr; TS[i+1]=T; mS[i+1]=m; qS[i+1]=hw*(T-Tr[0])
    p = mS*R*TS/V
    return dict(t=tv, m=mS, T=TS, p=p, Tr=TrS, q=qS, xr=xr,
                L=L, Nr=NR, n_cycles=NCYC, tc=TC, R=R, cp=CP, cv=CV,
                V=V, Aw=AW, p0=P0, T0=T0, Tinf=TINF, Tin=TIN, m0=M0,
                hw_true=hw, alpha_true=ALPHA, k_rock=K_ROCK,
                mdot_in=MDOT_IN, mdot_out=MDOT_OUT,
                Pi_h=hw*AW*TC/(M0*CV), Fo=ALPHA*TC/L**2)

if __name__ == '__main__':
    import sys
    d = solve()
    np.savez(sys.argv[1] if len(sys.argv)>1 else 'reference_regen.npz', **d)
