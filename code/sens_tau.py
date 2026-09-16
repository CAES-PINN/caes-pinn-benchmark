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
OBS={'A':(iA,-2.1),'B':(iB,-4.6)}     # انحراف مشاهده‌شدهٔ گام ۱۴

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

F=run([hwt],8,True)[0].astype(np.float64)
Om=run([hwt-0.5],8)[0]; Op=run([hwt+0.5],8)[0]
SV={w:(Op[:,ii]-Om[:,ii]).ravel().astype(np.float64) for w,(ii,_) in OBS.items()}

def proj(E,w):
    ii=OBS[w][0]; s=SV[w]; e=E[np.ix_(ridx,ii)].ravel()
    rr=np.sqrt((e**2).mean())
    return (e@s)/(s@s), rr, 100*abs(e@s)/(np.linalg.norm(e)*np.linalg.norm(s)+1e-30)

print('--- جاروب مقیاس میرایی زمانی (دامنهٔ طبیعی) ---')
print(' tau (h) |  dh_A   RMS_A  هم%  |  dh_B   RMS_B  هم%')
TAU=np.array([0.1,0.2,0.3,0.5,0.7,1.0,1.3,1.6,2.0,2.5]); rows=[]
for th in TAU:
    E=gaussian_filter1d(F,th*3600/dt,axis=1,mode='nearest')-F
    a=proj(E,'A'); b=proj(E,'B'); rows.append((a[0],b[0],a[1],b[1]))
    print(' %6.2f  | %+6.2f  %.4f  %4.1f  | %+6.2f  %.4f  %4.1f'%(
          th,a[0],a[1],a[2],b[0],b[1],b[2]))
rows=np.array(rows)
for j,w in enumerate('AB'):
    tgt=OBS[w][1]
    te=np.interp(-tgt,-rows[:,j],TAU); rq=np.interp(te,TAU,rows[:,2+j])
    print('  %s: انحراف %+.1f  ->  tau_موثر = %.2f h  با RMS لازم %.4f K'%(
          w,tgt,te,rq))

print('\n--- ترکیب فضایی+زمانی: آیا با هم انحراف مشاهده‌شده را می‌سازند؟ ---')
print(' sig_x(m)  tau(h) |   dh_A    dh_B   | RMS_A  RMS_B')
for sx in (0.0,0.128,0.257):
    for th in (0.5,1.0,1.5):
        E=gaussian_filter1d(gaussian_filter1d(F,max(sx/dx,1e-6),axis=0,mode='nearest'),
                            th*3600/dt,axis=1,mode='nearest')-F
        a=proj(E,'A'); b=proj(E,'B')
        print(' %7.3f  %6.2f | %+7.2f %+7.2f  | %.4f %.4f'%(sx,th,a[0],b[0],a[1],b[1]))
print('\nهدف: dh_A=-2.1 , dh_B=-4.6  با RMS در حد 0.19 K')
