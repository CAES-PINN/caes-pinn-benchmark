import numpy as np, time
P='/content/drive/MyDrive/CAES_PINN'
ref=np.load(P+'/reference_solution.npz',allow_pickle=True)
sen=np.load(P+'/sensor_data.npz',allow_pickle=True)
S=lambda d,k: float(np.asarray(d[k]).ravel()[0])

L=S(ref,'L'); Nr=int(S(ref,'Nr')); tc=S(ref,'tc'); R=S(ref,'R')
cp=S(ref,'cp'); cv=S(ref,'cv'); Aw=S(ref,'Aw'); T0=S(ref,'T0')
Tinf=S(ref,'Tinf'); Tin=S(ref,'Tin'); m0=S(ref,'m0')
hw_true=S(ref,'hw_true'); alpha=S(ref,'alpha_true'); k_rock=S(ref,'k_rock')
t=ref['t']; m_ref=ref['m']; T_ref=ref['T']; Tr_ref=ref['Tr']; xr=ref['xr']
dt=float(t[1]-t[0]); dx=float(xr[1]-xr[0]); Nt=len(t)
d=np.diff(m_ref)/dt; Min=np.where(d>1,d,0.); Mout=np.where(d<-1,-d,0.)

ridx=np.asarray(sen['rock_idx']).astype(int)
sig_r=S(sen,'sig_r'); A_c0=int(S(sen,'A_c0')); B_c0=int(S(sen,'B_c0'))
iA=np.round(sen['A_ts']/dt).astype(int)+int(A_c0*tc/dt)
iB=np.round(sen['B_ts']/dt).astype(int)+int(B_c0*tc/dt)
W={'A':(iA,sen['A_Tr_obs'],sen['A_Tr_true']),
   'B':(iB,sen['B_Tr_obs'],sen['B_Tr_true'])}

def run(hws,nsub):
    hw=np.asarray(hws,float); H=len(hw)
    h=dt/nsub; c1=alpha*h/dx**2; c2=2*alpha*h*hw/(k_rock*dx)
    T=np.full(H,T0); Tr=np.full((H,Nr),Tinf); m=m0
    Ts=np.empty((H,Nt)); Trs=np.empty((H,len(ridx),Nt))
    Ts[:,0]=T; Trs[:,:,0]=Tr[:,ridx]
    for i in range(Nt-1):
        mi,mo=Min[i],Mout[i]
        for _ in range(nsub):
            dT=(mi*(cp*Tin-cv*T)-mo*R*T-hw*Aw*(T-Tr[:,0]))/(m*cv)
            Tn=Tr.copy()
            Tn[:,1:-1]=Tr[:,1:-1]+c1*(Tr[:,2:]-2*Tr[:,1:-1]+Tr[:,:-2])
            Tn[:,0]=Tr[:,0]+2*c1*(Tr[:,1]-Tr[:,0])+c2*(T-Tr[:,0])
            Tn[:,-1]=Tinf
            T=T+h*dT; m=m+h*(mi-mo); Tr=Tn
        Ts[:,i+1]=T; Trs[:,:,i+1]=Tr[:,ridx]
    return Ts,Trs

def vertex(x,y):
    j=int(np.argmin(y)); j=min(max(j,1),len(x)-2)
    a=y[j-1]-2*y[j]+y[j+1]
    return x[j]-0.5*(x[j+1]-x[j-1])*(y[j+1]-y[j-1])/(4*a) if a>0 else x[j]

HW=np.arange(20.,40.01,0.5); j0=int(np.argmin(abs(HW-hw_true)))
rng=np.random.default_rng(0)
for nsub in (2,4,8):
    t0=time.time(); Ts,Trs=run(HW,nsub)
    e1=np.abs(Ts[j0]-T_ref).max(); e2=np.abs(Trs[j0]-Tr_ref[ridx,:]).max()
    print('\n===== nsub=%d  (%.0f s)  max|T-ref|=%.5f  max|Tr-ref|=%.5f K'%(
          nsub,time.time()-t0,e1,e2))
    for nm,(ii,obs,tru) in W.items():
        M=Trs[:,:,ii]
        c_obs=((M-obs[None])**2).sum(axis=(1,2))/sig_r**2
        c_tru=((M-tru[None])**2).sum(axis=(1,2))/sig_r**2
        sens=(Trs[j0+1,:,ii]-Trs[j0-1,:,ii])/(HW[j0+1]-HW[j0-1])
        sg=sig_r/np.sqrt((sens**2).sum())
        # توزیع تجربی برآوردگر روی 400 تحقق نوفه
        F=M.reshape(len(HW),-1); q=(F**2).sum(1)
        Y=tru.ravel()[None,:]+sig_r*rng.standard_normal((400,F.shape[1]))
        C=q[None,:]-2.0*(Y@F.T)
        est=np.array([vertex(HW,C[r]) for r in range(400)])
        print('  %s | argmin(نوفه‌دار)=%.2f | argmin(حقیقی)=%.2f | sigma_Fisher=%.2f'
              ' | MC: mean=%.2f std=%.2f'%(nm,vertex(HW,c_obs),vertex(HW,c_tru),
              sg,est.mean(),est.std()))
        print('       Delta_chi2(22)=%.1f  ->  %.1f sigma'%(
              c_obs[np.argmin(abs(HW-22))]-c_obs.min(),
              np.sqrt(max(c_obs[np.argmin(abs(HW-22))]-c_obs.min(),0))))
