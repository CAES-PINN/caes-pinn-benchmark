import numpy as np, json, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P   = '/content/drive/MyDrive/CAES_PINN'
ref = np.load(P+'/reference_solution.npz', allow_pickle=True)
sen = np.load(P+'/sensor_data.npz',        allow_pickle=True)
S   = lambda d,k: float(np.asarray(d[k]).ravel()[0])

L=S(ref,'L'); Nr=int(S(ref,'Nr')); tc=S(ref,'tc'); R=S(ref,'R')
cp=S(ref,'cp'); cv=S(ref,'cv'); Aw=S(ref,'Aw')
T0=S(ref,'T0'); Tinf=S(ref,'Tinf'); Tin=S(ref,'Tin')
m0=S(ref,'m0'); hw_true=S(ref,'hw_true')
alpha=S(ref,'alpha_true'); k_rock=S(ref,'k_rock')

t=ref['t']; m_ref=ref['m']; T_ref=ref['T']; Tr_ref=ref['Tr']; xr=ref['xr']
dt=float(t[1]-t[0]); dx=float(xr[1]-xr[0]); Nt=len(t)

# ---- بازیابی زمان‌بندی دبی از m(t) ----
d    = np.diff(m_ref)/dt
Min  = np.where(d> 1.0,  d, 0.0)
Mout = np.where(d<-1.0, -d, 0.0)
n1   = int(tc/dt)
print('dt=%.1f s | dx=%.5f m | Nt=%d'%(dt,dx,Nt))
print('زمان‌بندی چرخهٔ ۱: شارژ %.2f h | دشارژ %.2f h | سکون %.2f h'%(
      (Min[:n1]>0).sum()*dt/3600,(Mout[:n1]>0).sum()*dt/3600,
      ((Min[:n1]==0)&(Mout[:n1]==0)).sum()*dt/3600))
print('mdot_in=%.1f  mdot_out=%.1f  (Fourier r=%.4f)'%(
      Min.max(),Mout.max(),alpha*dt/dx**2))

ridx=np.asarray(sen['rock_idx']).astype(int)
sig_T=S(sen,'sig_T'); sig_r=S(sen,'sig_r')
A_c0=int(S(sen,'A_c0')); B_c0=int(S(sen,'B_c0'))
iA=np.round(sen['A_ts']/dt).astype(int)+int(A_c0*tc/dt)
iB=np.round(sen['B_ts']/dt).astype(int)+int(B_c0*tc/dt)

# ---- حل مستقیم، برداری روی همهٔ h_w ها ----
def run(hws, nsub=2):
    hw=np.asarray(hws,float); H=len(hw)
    h=dt/nsub; c1=alpha*h/dx**2; c2=2*alpha*h*hw/(k_rock*dx)
    T=np.full(H,T0); Tr=np.full((H,Nr),Tinf); m=m0
    Ts=np.empty((H,Nt)); Trs=np.empty((H,len(ridx),Nt))
    Ts[:,0]=T; Trs[:,:,0]=Tr[:,ridx]
    for i in range(Nt-1):
        mi,mo=Min[i],Mout[i]
        for _ in range(nsub):
            Ts0=Tr[:,0]
            dT=(mi*(cp*Tin-cv*T)-mo*R*T-hw*Aw*(T-Ts0))/(m*cv)
            Tn=Tr.copy()
            Tn[:,1:-1]=Tr[:,1:-1]+c1*(Tr[:,2:]-2*Tr[:,1:-1]+Tr[:,:-2])
            Tn[:,0]   =Tr[:,0]+2*c1*(Tr[:,1]-Tr[:,0])+c2*(T-Tr[:,0])
            Tn[:,-1]  =Tinf
            T=T+h*dT; m=m+h*(mi-mo); Tr=Tn
        Ts[:,i+1]=T; Trs[:,:,i+1]=Tr[:,ridx]
    return Ts,Trs

# ---- اعتبارسنجی با h_w = 30 ----
t0=time.time(); Tv,Rv=run([hw_true]); print('\nزمان یک حل: %.1f s'%(time.time()-t0))
print('=== اعتبارسنجی (باید < 0.05 K) ===')
print('  max|T - T_ref|        = %.4f K'%np.abs(Tv[0]-T_ref).max())
print('  max|Tr - Tr_ref|@sens = %.4f K'%np.abs(Rv[0]-Tr_ref[ridx,:]).max())
print('  max|A_Tr - A_Tr_true| = %.4f K'%np.abs(Rv[0][:,iA]-sen['A_Tr_true']).max())
print('  max|B_Tr - B_Tr_true| = %.4f K'%np.abs(Rv[0][:,iB]-sen['B_Tr_true']).max())
print('  max|A_T  - A_T_true|  = %.4f K'%np.abs(Tv[0][iA]-sen['A_T_true']).max())

# ---- جاروب h_w ----
HW=np.arange(20.0,40.01,0.5); Ts,Trs=run(HW)
j0=int(np.argmin(abs(HW-hw_true)))
out={'HW':HW}
for nm,ii,obs in [('A',iA,sen['A_Tr_obs']),('B',iB,sen['B_Tr_obs'])]:
    D   = Trs[:,:,ii]-Trs[j0:j0+1,:,ii]
    rms = np.sqrt((D**2).mean(axis=(1,2))); mx=np.abs(D).max(axis=(1,2))
    chi = ((Trs[:,:,ii]-obs[None])**2).sum(axis=(1,2))/sig_r**2
    N   = obs.size
    print('\n=== پنجرهٔ %s  (N=%d سنجش سنگ، sigma_r=%.2f K) ==='%(nm,N,sig_r))
    print(' h_w | RMS dev vs 30 |  max dev  |  chi2 - chi2min')
    for a,b,c,e in zip(HW,rms,mx,chi-chi.min()):
        star=' <-- min' if abs(e)<1e-9 else ''
        if a%1==0: print(' %4.1f |   %.5f K   | %.5f K |  %8.2f%s'%(a,b,c,e,star))
    jm=int(np.argmin(chi))
    sens=(Trs[j0+1,:,ii]-Trs[j0-1,:,ii])/(HW[j0+1]-HW[j0-1])
    sg  = sig_r/np.sqrt((sens**2).sum())
    ok  = HW[chi<=chi.min()+1.0]
    print('  argmin chi2      : h_w = %.1f'%HW[jm])
    print('  بازهٔ 1-sigma      : [%.1f , %.1f]'%(ok.min(),ok.max()))
    print('  sigma فیشر        : %.2f W/m2K'%sg)
    print('  حساسیت میانگین    : %.3g K per W/m2K'%np.abs(sens).mean())
    print('  RMS(22 vs 30)     : %.5f K  <->  نوفهٔ سنجش %.2f K'%(
          rms[np.argmin(abs(HW-22))],sig_r))
    out['rms_'+nm]=rms; out['max_'+nm]=mx; out['chi2_'+nm]=chi

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for nm,s in [('A','o-'),('B','s--')]:
    ax[0].plot(HW,out['chi2_'+nm]-out['chi2_'+nm].min(),s,label='window '+nm)
    ax[1].plot(HW,out['rms_'+nm],s,label='window '+nm)
ax[0].axhline(1,color='r',ls=':'); ax[0].axvline(30,color='k',ls=':')
ax[0].set_xlabel('$h_w$'); ax[0].set_ylabel(r'$\chi^2-\chi^2_{min}$'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].axhline(sig_r,color='r',ls='-.'); ax[1].axvline(30,color='k',ls=':')
ax[1].set_yscale('log'); ax[1].set_xlabel('$h_w$'); ax[1].set_ylabel('RMS dev vs true (K)')
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(P+'/fig7_profile_likelihood.png',dpi=160)
np.savez(P+'/sens_profile.npz',**out)
print('\nذخیره شد: fig7_profile_likelihood.png و sens_profile.npz')
