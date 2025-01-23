import numpy as np
import math
from scipy.special import *
import matplotlib.pyplot as plt
import csv
#WPTのSP方式もしくはDRSSTCの負荷特性をシミュレーション.
#参考:
#米田 昇平, 木船 弘康, 「共振コンデンサを直列または並列に接続した電圧源駆動非接触給電回路の共振周波数と負荷電圧」, 電気学会論文誌Ｄ（産業応用部門誌）, 2020, 140 巻, 9 号, p. 642-650, https://doi.org/10.1541/ieejias.140.642.
#もしくは以下の文献なら無料でアクセス可能
#米田 昇平, 木船 弘康, 「共振周波数追従制御を適用した水中探査機向け非接触給電システムの負荷電圧特性の検討」

def tesla_SSTC_lumped_Fmatrix(Ls,Cs,n,Rp,Rls,k,Omega):
    k2=k**2
    F=np.array([[1,Rp],[0,1]])@np.array([[1/n,0],[0,n]])@np.array([[1,0],[1/(Omega*Ls*1j+Rls)/k2,1]])@np.array([[1,(Omega*Ls*1j+Rls)*(1-k2)],[0,1]])@np.array([[1,0],[Omega*Cs*1j,1]])
    return F
def tesla_SSTC_lumped_calcu(F,Rs):
    G=1/(F[0][0]+F[0][1]/Rs)
    eta=1/np.real((Rs*F[0][0]+F[0][1])*np.conj(F[1][0]+F[1][1]/Rs))
    return G,eta
def tesla_DRSSTC_optim_func():
    #tesla coil lumped model charactaristics
    n=k*math.sqrt(Ls/Lp)
    Cp=1/(Omega**2*Lp*(1-k**2))
    F_SSTC=tesla_SSTC_lumped_Fmatrix(Ls,Cs,n,Rlp,Rls,k,Omega2)
    F_DRSSTC=np.array([[1,1/(Omega2*Cp*1j)],[0,1]])@F_SSTC
    G,eta=tesla_SSTC_lumped_calcu(F_DRSSTC,Rs)
    return G,eta

Lp=10e-6#一次コイルのインダクタンス[H].
Ls=10e-3#二次コイルのインダクタンス[H].
Rlp=0.5#一次コイルの抵抗[Ohm].
Rls=10#二次コイルの抵抗[Ohm].
k=0.2#一次・二次間の結合係数[-].
f=200e3#動作周波数[Hz].

Omega=2*math.pi*f
Cs=1/(Omega**2*Ls)

imax=1000#point数.
matRs=np.logspace(3,7,imax)#負荷抵抗の範囲はここで変える.logspaceの引数がn,mなら10^nから10^mになる.

G_ans=[]
eta_ans=[]
matf2=[]
matRs2=[]
matf3=[]
G_lossless=[]
G_lossless2=[]
G_lossless3=[]
Rmax=math.sqrt((1-k**2)*Ls/Cs)/k#=Rssp.
for Rs in matRs:
    if Rs>Rmax:
        Asp=1-k**2/2-(1-k**2)*Ls/(2*Cs*Rs**2)
        Omega2=math.sqrt((Asp-math.sqrt(Asp**2-(1-k**2)**2))/(Ls*Cs*(1-k**2)))
        G_lossless.append(Rs/Rmax/k*np.sqrt(Ls/Lp))
        matf2.append(Omega2/(2*np.pi))
        matRs2.append(Rs)
        matf3.append((math.sqrt((Asp+math.sqrt(Asp**2-(1-k**2)**2))/(Ls*Cs*(1-k**2))))/(2*np.pi))
        G_lossless3.append(Rs/Rmax/k*np.sqrt(Ls/Lp))
    else:
        Omega2=Omega
        G_lossless.append(1/k*np.sqrt(Ls/Lp))
    G_lossless2.append(1/k*np.sqrt(Ls/Lp))
    G,eta=tesla_DRSSTC_optim_func()
    G_ans.append(G)
    eta_ans.append(eta)
#print(1/(math.sqrt(Ls*Cs*(1+k))*math.pi*2))

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
ax[0,0].set_title("Lossy and Lossless Gain and Power Efficiency")
ax[0,0].plot(matRs,G_lossless,"r--")
ax[0,0].plot(matRs,np.abs(G_ans),"k-")
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel("Gain [-]")
ax[1,0].plot(matRs,eta_ans,"k-")
ax[1,0].set_xscale('log')
ax[1,0].set_ylabel("Power Efficiency [-]")
ax[1,0].set_xlabel("Load Resistance [Ohm]")

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
ax[0,0].set_title("Lossy and Lossless Gain and Resonant Frequency")
ax[0,0].plot(matRs,G_lossless,"r--")
ax[0,0].plot(matRs,np.abs(G_ans),"k-")
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel("Gain [-]")
ax[1,0].plot([matRs[0],matRs2[0]],[f*1e-3,f*1e-3],"k-")
ax[1,0].plot(matRs2,np.array(matf2)*1e-3,"k--")
ax[1,0].plot(matRs2,np.array(matf3)*1e-3,"k-")
ax[1,0].set_xscale('log')
ax[1,0].set_ylabel("Resonant Frequency [kHz]")
ax[1,0].set_xlabel("Load Resistance [Ohm]")

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
ax[0,0].set_title("Lossless Gain and Resonant Frequency")
ax[0,0].plot(matRs,G_lossless2,"k-")
ax[0,0].plot(matRs2,G_lossless3,"r--")
ax[0,0].plot([matRs[0],matRs[0]],[10,10],"k-",alpha=0)
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel("Gain [-]")
ax[1,0].plot([matRs[0],matRs[-1]],[f*1e-3,f*1e-3],"k-")
ax[1,0].plot(matRs2,np.array(matf2)*1e-3,"r--")
ax[1,0].plot(matRs2,np.array(matf3)*1e-3,"r--")
ax[1,0].plot([matRs[0],matRs[1]],[250,150],"k-",alpha=0)
ax[1,0].set_xscale('log')
ax[1,0].set_ylabel("Resonant Frequency [kHz]")
ax[1,0].set_xlabel("Load Resistance [Ohm]")

plt.show()