import numpy as np
from scipy.ndimage import gaussian_filter1d
P='/content/drive/MyDrive/CAES_PINN'
ref=np.load(P+'/reference_solution.npz',allow_pickle=True)
sen=np.load(P+'/sensor_data.npz',allow_pickle=True)
S=lambda d,k: float(np.asarray(d[k]).ravel()[0])
Nr=int(S(ref,'Nr')); tc=S(ref,'tc'); R=S(ref,'R'); cp=S(ref,'cp'); cv=S(ref,'cv')
Aw=S(ref,'Aw'); T0=S(ref,'T0'); Tinf=S(ref,'Tinf'); Tin=S(ref,'Tin'); m0=S(ref,'m0')
hwt=S(ref,'hw_true'); alpha=S(ref,'alpha_true'); k_rock=S(ref,'k_rock')
t=ref['t']; xr=ref['xr']; dt=float(t[1]-t[0]); dx=float(xr[1]-xr[0]); Nt=len(t)
d=np.diff(ref['m'])/dt; Min=np.where(d>1,d,0.); Mout=np.where(d<-1,-d,0.)
ridx=np.asarray(sen['rock_idx']).astype(int)
iA=np.round(sen['A_ts']/dt).astype(int)+int(S(sen,'A_c0')*tc/dt)
iB=np.round(sen['B_ts']/dt).astype(int)+int(S(sen,'B_c0')*tc/dt)
W={'A':(iA,27.9),'B':(iB,25.4)}
rmsP=0.189

def run(hws,nsub=8,full=False):
    hw=np.asarray(hws,float); H=len(hw); h=dt/nsub
    c1=alpha*h/dx**2; c2=2*alpha*h*hw/(k_rock*dx)
    T=np.full(H,T0); Tr=np.full((H,Nr),Tinf); m=m0
    O=np.empty((H,Nr if full else len(ridx),Nt),dtype=np.float32)
    O[:,:,0]=Tr if full else Tr[:,ridx]
    for i in range(Nt-1):
        mi,mo=Min[i],Mout[i]
        for _ in range(nsub):
            dT=(mi*(cp*Tin-cv*T)-mo*R*T-hw*Aw*(T-Tr[:,0]))/(m*cv)
            Tn=Tr.copy()
            Tn[:,1:-1]=Tr[:,1:-1]+c1*(Tr[:,2:]-2*Tr[:,1:-1]+Tr[:,:-2])
            Tn[:,0]=Tr[:,0]+2*c1*(Tr[:,1]-Tr[:,0])+c2*(T-Tr[:,0]); Tn[:,-1]=Tinf
            T=T+h*dT; m=m+h*(mi-mo); Tr=Tn
        O[:,:,i+1]=Tr if full else Tr[:,ridx]
    return O

Tr8=run([hwt],8,True)[0].astype(np.float64)
Tr2=run([hwt],2,True)[0].astype(np.float64)
Om=run([hwt-0.5],8)[0]; Op=run([hwt+0.5],8)[0]
print('عمق نفوذ حرارتی sqrt(alpha*tc) = %.3f m  = %.1f سلول شبکه\n'%(
      np.sqrt(alpha*tc), np.sqrt(alpha*tc)/dx))

cases=[]
for n in (1,2,4,8,12):
    cases.append(('هموارسازی فضایی  %.3f m'%(n*dx),
                  gaussian_filter1d(Tr8,n,axis=0,mode='nearest')-Tr8))
for n in (5,15,40,90):
    cases.append(('هموارسازی زمانی  %.2f h'%(n*dt/3600),
                  gaussian_filter1d(Tr8,n,axis=1,mode='nearest')-Tr8))
cases.append(('خطای گام درشت (nsub=2)', Tr2-Tr8))

hdr='%-26s %8s | %7s %7s %6s | %7s %7s %6s'
print(hdr%('نوع خطا','RMS خام','dh_A','خام_A','هم‌A%','dh_B','خام_B','هم‌B%'))
for nm,E in cases:
    row=[]; r0=[]
    for w,(ii,hp) in W.items():
        s=(Op[:,ii]-Om[:,ii]).ravel().astype(np.float64)
        e=E[np.ix_(ridx,ii)].ravel()
        rr=np.sqrt((e**2).mean())
        dh_raw=(e@s)/(s@s)
        dh=dh_raw*rmsP/rr if rr>0 else 0.
        al=100*abs(e@s)/(np.linalg.norm(e)*np.linalg.norm(s)+1e-30)
        row+= [dh,dh_raw,al]; r0.append(rr)
    print('%-26s %8.4f | %+7.2f %+7.3f %6.1f | %+7.2f %+7.3f %6.1f'%(
          nm,r0[0],*row))
print('\n(dh = انحراف h_w اگر خطا به RMS=%.3f K مقیاس شود؛ خام = با دامنهٔ طبیعی خودش)'%rmsP)
