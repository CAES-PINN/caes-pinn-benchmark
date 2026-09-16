import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P='/content/drive/MyDrive/CAES_PINN'
ref=np.load(P+'/reference_solution.npz',allow_pickle=True)
sen=np.load(P+'/sensor_data.npz',allow_pickle=True)
S=lambda d,k: float(np.asarray(d[k]).ravel()[0])
Nr=int(S(ref,'Nr')); tc=S(ref,'tc'); R=S(ref,'R'); cp=S(ref,'cp'); cv=S(ref,'cv')
Aw=S(ref,'Aw'); T0=S(ref,'T0'); Tinf=S(ref,'Tinf'); Tin=S(ref,'Tin'); m0=S(ref,'m0')
hwt=S(ref,'hw_true'); alpha=S(ref,'alpha_true'); k_rock=S(ref,'k_rock')
t=ref['t']; xr=ref['xr']; dt=float(t[1]-t[0]); dx=float(xr[1]-xr[0]); Nt=len(t)
d=np.diff(ref['m'])/dt; Min=np.where(d>1,d,0.); Mout=np.where(d<-1,-d,0.)
ridx=np.asarray(sen['rock_idx']).astype(int); sig_r=S(sen,'sig_r')
iA=np.round(sen['A_ts']/dt).astype(int)+int(S(sen,'A_c0')*tc/dt)
iB=np.round(sen['B_ts']/dt).astype(int)+int(S(sen,'B_c0')*tc/dt)
W={'A':(iA,sen['A_Tr_obs'],27.9),'B':(iB,sen['B_Tr_obs'],25.4)}

def run(hws,nsub=8):
    hw=np.asarray(hws,float); H=len(hw); h=dt/nsub
    c1=alpha*h/dx**2; c2=2*alpha*h*hw/(k_rock*dx)
    T=np.full(H,T0); Tr=np.full((H,Nr),Tinf); m=m0
    O=np.empty((H,len(ridx),Nt)); O[:,:,0]=Tr[:,ridx]
    for i in range(Nt-1):
        mi,mo=Min[i],Mout[i]
        for _ in range(nsub):
            dT=(mi*(cp*Tin-cv*T)-mo*R*T-hw*Aw*(T-Tr[:,0]))/(m*cv)
            Tn=Tr.copy()
            Tn[:,1:-1]=Tr[:,1:-1]+c1*(Tr[:,2:]-2*Tr[:,1:-1]+Tr[:,:-2])
            Tn[:,0]=Tr[:,0]+2*c1*(Tr[:,1]-Tr[:,0])+c2*(T-Tr[:,0]); Tn[:,-1]=Tinf
            T=T+h*dT; m=m+h*(mi-mo); Tr=Tn
        O[:,:,i+1]=Tr[:,ridx]
    return O

HW=np.arange(20.,40.01,0.5); j0=int(np.argmin(abs(HW-hwt)))
O=run(np.r_[HW,hwt-0.5,hwt+0.5]); Og=O[:len(HW)]; Om,Op=O[-2],O[-1]
rng=np.random.default_rng(1); rmsP=0.189; res={}
print('RMS خطای حل PINN فرضی: %.3f K   |  N=%d'%(rmsP,sen['A_Tr_obs'].size))
for nm,(ii,obs,hpinn) in W.items():
    s=((Op[:,ii]-Om[:,ii])/1.0).ravel()           # بردار حساسیت dT/dh
    ns=np.sqrt((s**2).sum()); sg=sig_r/ns
    xs=xr[ridx]; ts=t[ii]
    Xn=(xs-xs.min())/np.ptp(xs); Tn2=(ts-ts.min())/np.ptp(ts)
    B=np.array([np.cos(p*np.pi*Xn)[:,None]*np.cos(q*np.pi*Tn2)[None,:]
                for p in range(4) for q in range(4)]).reshape(16,-1)
    out={}
    for lab,E in [('سفید (ناهمبسته)',rng.standard_normal((3000,s.size))),
                  ('هموار (۱۶ مود)',rng.standard_normal((3000,16))@B)]:
        E=E/np.sqrt((E**2).mean(1,keepdims=True))*rmsP
        dh=(E@s)/ns**2; out[lab]=dh
        print('  %s | %-18s -> انحراف: میانگین %+.2f  std %.2f  |P95|=%.2f واحد'%(
              nm,lab,dh.mean(),dh.std(),np.percentile(abs(dh),95)))
    worst=rmsP*np.sqrt(s.size)/ns
    c=abs(hpinn-hwt)*ns/(rmsP*np.sqrt(s.size))
    print('  %s | sigma_CR=%.2f | بدترین حالت (هم‌راستای کامل)=%.1f واحد'%(nm,sg,worst))
    print('  %s | PINN گام۱۴ داد %.1f  ==>  هم‌راستایی لازم فقط %.1f%%\n'%(nm,hpinn,100*c))
    res[nm]=(out,sg)

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
for nm,(ii,obs,_) in W.items():
    ch=((Og[:,:,ii]-obs[None])**2).sum(axis=(1,2))/sig_r**2
    ax[0].plot(HW,ch-ch.min(),'o-' if nm=='A' else 's--',label='window '+nm)
ax[0].axhline(1,color='r',ls=':'); ax[0].axvline(30,color='k',ls=':')
ax[0].set_ylim(0,40); ax[0].set_xlabel('$h_w$  (W/m$^2$K)')
ax[0].set_ylabel(r'$\chi^2-\chi^2_{\min}$'); ax[0].legend(); ax[0].grid(alpha=.3)
for lab,st in zip(res['A'][0],['-','--']):
    ax[1].hist(30+res['A'][0][lab],bins=60,histtype='step',ls=st,label=lab)
ax[1].axvline(30,color='k',ls=':'); ax[1].axvline(27.9,color='r',ls='-.',label='PINN گام۱۴')
ax[1].set_xlabel(r'$\hat h_w$'); ax[1].set_ylabel('count'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(P+'/fig7_detection_limit.png',dpi=160)
np.savez(P+'/bias_analysis.npz',HW=HW,**{'dh_'+k:v[0]['سفید (ناهمبسته)'] for k,v in res.items()})
print('ذخیره شد: fig7_detection_limit.png و bias_analysis.npz')
