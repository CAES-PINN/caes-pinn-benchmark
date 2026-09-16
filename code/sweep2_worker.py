import os, sys, json, glob, time
import numpy as np, torch, torch.nn as nn

P = '/content/drive/MyDrive/CAES_PINN'
D = P + '/sweep2'; os.makedirs(D, exist_ok=True)
TASKS = [(10.,24.),(10.,36.),(1.,24.),(1.,36.),(100.,24.),(100.,36.)]
AIR = 'A_T_true'
N_WARM, N_ADAM, N_LB, SEED = 2000, 7500, 2000, 0
HLO, HHI = 5.0, 100.0
tg = lambda t: 'lbc%05.1f_h%04.1f' % t
done = sorted(os.path.basename(f)[:-5] for f in glob.glob(D+'/*.json'))
todo = [t for t in TASKS if tg(t) not in done]
print('done:', done); print('todo:', [tg(t) for t in todo], flush=True)
if not todo: print('>>> ALL DONE'); sys.exit(0)
WBC, HINI = todo[0]

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ref = np.load(P+'/reference_solution.npz', allow_pickle=True)
sen = np.load(P+'/sensor_data.npz', allow_pickle=True)
G = lambda d,n: float(np.asarray(d[n]).ravel()[0])
L, k, al = G(ref,'L'), G(ref,'k_rock'), G(ref,'alpha_true')
tcy, Tinf = G(ref,'tc'), G(ref,'Tinf')
Fo = al*tcy/L**2
Xs = np.asarray(sen['x_rock'],float)/L
ts = np.asarray(sen['A_ts'],float)
Tro = np.asarray(sen['A_Tr_obs'],float)
DT = float(np.ptp(np.asarray(sen['A_T_obs'],float)))
tau = (ts-ts[0])/tcy; tm = float(tau[-1])
THo = (Tro-Tinf)/DT
T = lambda a: torch.tensor(np.asarray(a,np.float32), device=dev)

class Net(nn.Module):
    def __init__(s, w=64, d=5):
        super().__init__()
        ly = [nn.Linear(2,w), nn.Tanh()]
        for _ in range(d-1): ly += [nn.Linear(w,w), nn.Tanh()]
        s.f = nn.Sequential(*ly, nn.Linear(w,1))
    def forward(s, X, t): return s.f(torch.cat([X, t/tm], 1))

torch.manual_seed(SEED); np.random.seed(SEED)
THa = (np.asarray(sen[AIR],float)-Tinf)/DT
net = Net().to(dev)
s0  = float(np.log((HINI-HLO)/(HHI-HINI)))
sp  = nn.Parameter(torch.tensor(s0, device=dev))
hfun = lambda: HLO + (HHI-HLO)*torch.sigmoid(sp)

Xc = T(np.random.rand(5000,1)**2); tt = T(np.random.rand(5000,1)*tm)
Xc.requires_grad_(True); tt.requires_grad_(True)
X0 = T(np.linspace(0,1,300)[:,None]**2); tz = T(np.zeros((300,1)))
tb = T(np.linspace(0,tm,800)[:,None]); tb.requires_grad_(True)
Zb = T(np.zeros((800,1))); Zb.requires_grad_(True)
Ob = T(np.ones((800,1)));  Ob.requires_grad_(True)
THab = T(np.interp(tb.detach().cpu().numpy().ravel(), tau, THa)[:,None])
Xd = T(np.tile(Xs,(ts.size,1)).reshape(-1,1))
td = T(np.repeat(tau, Xs.size).reshape(-1,1))
THd = T(THo.T.reshape(-1,1))
d1 = lambda y,v: torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]

def ls():
    eps = k/(hfun()*L)
    q = net(Xc,tt); qx = d1(q,Xc); qw = net(Zb,tb)
    return (((d1(q,tt) - Fo*d1(qx,Xc))**2).mean(),
            ((-eps*d1(qw,Zb) - (THab - qw))**2).mean(),
            ((d1(net(Ob,tb),Ob))**2).mean(),
            ((net(X0,tz))**2).mean(),
            ((net(Xd,td) - THd)**2).mean())
W = [1., WBC, 1., 10., 100.]
tot = lambda l: sum(w*v for w,v in zip(W,l))

t0 = time.time()
print('>>> lbc=%.1f h_init=%.1f (%s)' % (WBC,HINI,dev), flush=True)
o1 = torch.optim.Adam(net.parameters(), lr=1e-3)
for i in range(N_WARM):
    o1.zero_grad(); v = tot(ls()); v.backward(); o1.step()
    if (i+1) % 1000 == 0:
        print('  warm %5d J=%.4e rms=%.4f' % (i+1,float(v),DT*float(ls()[4])**.5), flush=True)
o2 = torch.optim.Adam([{'params':net.parameters(),'lr':1e-3},
                       {'params':[sp],'lr':3e-3}])
for i in range(N_ADAM):
    o2.zero_grad(); v = tot(ls()); v.backward(); o2.step()
    if (i+1) % 1500 == 0:
        print('  adam %5d J=%.4e h=%.4f rms=%.4f' % (i+1,float(v),float(hfun()),
              DT*float(ls()[4])**.5), flush=True)
lb = torch.optim.LBFGS(list(net.parameters())+[sp], max_iter=N_LB,
                       tolerance_grad=1e-11, line_search_fn='strong_wolfe')
def cl():
    lb.zero_grad(); v = tot(ls())
    if torch.isfinite(v): v.backward()
    return v
lb.step(cl)

l = [float(x) for x in ls()]
h_hat = float(hfun())
qw = net(Zb,tb); gx = d1(qw,Zb)
g = gx.detach().cpu().numpy().ravel()
dn = (THab - qw).detach().cpu().numpy().ravel()
sgd, sdd, sgg = float((g*dn).sum()), float((dn*dn).sum()), float((g*g).sum())
h_imp = -(k/L)*sgd/sdd
h_eps = -(k/L)*sgg/sgd
r2    = sgd**2/(sgg*sdd)
R = DT*l[4]**0.5
print('  h_hat=%.4f h_implied=%.4f h_epsform=%.4f r2=%.5f RMS=%.4f K'
      % (h_hat,h_imp,h_eps,r2,R), flush=True)
json.dump(dict(lbc=WBC, h_init=HINI, h_hat=h_hat, h_implied=h_imp,
               h_epsform=h_eps, r2=r2, rms=R, comp=l, W=W, air=AIR,
               seed=SEED, hlo=HLO, hhi=HHI, n_warm=N_WARM,
               sec=int(time.time()-t0)),
          open(D+'/%s.json' % tg((WBC,HINI)),'w'))
print('>>> saved. %d task(s) left.' % (len(todo)-1), flush=True)
