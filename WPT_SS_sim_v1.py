import numpy as np
import math
import cmath
from scipy.special import *
import matplotlib.pyplot as plt

def nagaoka(len,dia):
    q1=dia/len
    q2=np.sqrt(1/(1+1/q1**2))
    KK=ellipk(q2**2)
    EE=ellipe(q2**2)
    Kn=4/(3*math.pi*np.sqrt(1-q2**2))*((1/q2**2-1)*KK-(1/q2**2-2)*EE-q2)
    return Kn
def coupling_coeff_calcu(ra,rb,lena,lenb,leng,dw_pu,N,MYU):
    #                lena     leng
    #           <----------> <-------->
    #           ------------
    #           |        <------------> lenb
    #           |        --------------
    #     La(ra)| Lb(rb)|             |
    #           |        --------------
    #           |           
    #           ------------

    #dw=len/Na*dw_pu
    #lena>lenb
    Na=N
    Nb=N
    dw=lena/Na*dw_pu#wire dia
    #x coordinate of La,Lb
    matxa=np.linspace(0,lena,Na)
    matxb=leng+np.linspace(0,lenb,Nb)

    matM0=[]
    for i in range(Nb):
        h=np.abs(matxa[0]-matxb[i])
        lambda_origin=2*math.sqrt(ra*rb/((ra+rb)**2+h**2))
        KK=ellipk(lambda_origin**2)
        EE=ellipe(lambda_origin**2)
        matM0.append(MYU*np.sqrt(ra*rb)*((2/lambda_origin-lambda_origin)*KK-2/lambda_origin*EE))

    matmult=np.linspace(Nb,1,Nb)
    Lm=np.sum(matM0*matmult*2)-matM0[0]*Nb

    #x coordinate of La,Lb
    matxa=np.linspace(0,lena,Na)
    matxb=leng+np.linspace(0,lenb,Nb)

    #Calc La
    dw=lena/Na*dw_pu#wire dia
    matM22=[]
    for i in range(Na):
        h=np.abs(matxa[0]-matxa[i])
        lambda_origin=2*np.sqrt(ra*ra/((ra+ra)**2+h**2))
        if h==0:
            matM22.append(MYU*ra*(np.log(8*ra/dw)-7/4))
        else:
            KK=ellipk(lambda_origin**2)
            EE=ellipe(lambda_origin**2)
            matM22.append(MYU*ra*((2/lambda_origin-lambda_origin)*KK-2/lambda_origin*EE))

    matmult=np.linspace(Na,1,Na)
    La0=np.sum(matM22*matmult*2)-matM22[0]*Na

    #Calc Lb
    dw=lenb/Nb*dw_pu#wire dia
    matM32=[]
    for i in range(Nb):
        h=np.abs(matxb[0]-matxb[i])
        lambda_origin=2*np.sqrt(rb*rb/((rb+rb)**2+h**2))
        if h==0:
            matM32.append(MYU*rb*(np.log(8*rb/dw)-7/4))
        else:
            KK=ellipk(lambda_origin**2)
            EE=ellipe(lambda_origin**2)
            matM32.append(MYU*rb*((2/lambda_origin-lambda_origin)*KK-2/lambda_origin*EE))

    matmult=np.linspace(Nb,1,Nb)
    Lb0=np.sum(matM32*matmult*2)-matM32[0]*Nb

    k=Lm/math.sqrt(La0*Lb0)#Calc couppling coeff
    return k
def dowell_proximity_calcu(f,d,myu,sigma,eta_coil,m):
    alpha=cmath.sqrt(1j*2*math.pi*f*myu*eta_coil*sigma)
    M=alpha*d*cmath.tanh(alpha*d)
    D=2*alpha*d/cmath.tanh(alpha*d/2)
    M_prime=np.real(M)
    D_prime=np.real(D)
    R_ratio=M_prime+(m**2-1)*D_prime/3
    return R_ratio
def SS_lumped_Fmatrix(Ls,Cs,Cp,n,Rp,Rls,k,Omega):
    k2=k**2
    F=np.array([[1,1/(Omega*Cp*1j)],[0,1]])@np.array([[1,Rp],[0,1]])@np.array([[1/n,0],[0,n]])@np.array([[1,0],[1/(Omega*Ls*1j+Rls)/k2,1]])@np.array([[1,(Omega*Ls*1j+Rls)*(1-k2)],[0,1]])@np.array([[1,1/(Omega*Cs*1j)],[0,1]])
    return F
def F_lumped_calcu(F,Rs):
    G=1/(F[0][0]+F[0][1]/Rs)
    eta=1/np.real((Rs*F[0][0]+F[0][1])*np.conj(F[1][0]+F[1][1]/Rs))
    return G,eta

myu=4*math.pi*1e-7
eps=8.854e-12
cv=2.998e8
sigma=5.8e7#S/m , 銅の導電率.

N_sec=50
len_sec=10e-3
dia_sec=30e-3
eta_sec=0.2
wire_dia_sec=0.7e-3
layer_sec=4

N_pri=20
len_pri=10e-3
dia_pri=30e-3
eta_pri=0.2
wire_dia_pri=0.7e-3
layer_pri=2

fr=100e3

gap=50e-3

dw_pu=0.1#dw=len/Na*dw_pu when calcu k

dowell_coef=1#Rac=Rdc * dowell_proximity_calcu(any param) * dowell_coef , 実際のAC抵抗と近づけるための補正係数.

k=coupling_coeff_calcu(1/2,dia_sec/dia_pri/2,len_pri/dia_pri,len_sec/dia_pri,gap,dw_pu,200,myu)

Lp=myu*math.pi*dia_sec**2/4*N_pri**2/len_pri*nagaoka(len_pri,dia_pri)
Ls=myu*math.pi*dia_sec**2/4*N_sec**2/len_sec*nagaoka(len_sec,dia_sec)

Rdc_pri=N_pri*dia_pri*math.pi/(math.pi*(wire_dia_pri/2)**2)/sigma
k_Rlp=dowell_coef*dowell_proximity_calcu(fr,wire_dia_pri,myu,sigma,eta_pri,layer_pri)
Rlp=Rdc_pri*k_Rlp

Rdc_sec=N_sec*dia_sec*math.pi/(math.pi*(wire_dia_sec/2)**2)/sigma
k_Rls=dowell_coef*dowell_proximity_calcu(fr,wire_dia_sec,myu,sigma,eta_sec,layer_sec)
Rls=Rdc_sec*k_Rls

Cp=1/((2*math.pi*fr)**2*Lp)
Cs=1/((2*math.pi*fr)**2*Ls)
n=math.sqrt(Ls/Lp)*k
Rss=math.sqrt(2*Ls*(1-math.sqrt(1-k**2))/Cs)

print("k=",end="")
print(k,end=", ")
print("Lp=",end="")
print(Lp,end=", ")
print("Ls=",end="")
print(Ls,end=", ")
print("Cp=",end="")
print(Cp,end=", ")
print("Cs=",end="")
print(Cs,end=", ")
print("Rlp=",end="")
print(Rlp,end=", ")
print("Rls=",end="")
print(Rls)
print(Rdc_pri,Rdc_sec)

imax=1000
matRs=np.logspace(0,3,imax)

G_ans_SS=[]
eta_ans_SS=[]
for Rs in matRs:
    if Rs<Rss:
        Omega=math.sqrt((1-Cs*Rs**2/(2*Ls)-math.sqrt((1-Cs*Rs**2/(2*Ls))**2-1+k**2))/(Ls*Cs*(1-k**2)))
    else:
        Omega=2*math.pi*fr
    F=SS_lumped_Fmatrix(Ls,Cs,Cp,n,Rlp,Rls,k,Omega)
    G,eta=F_lumped_calcu(F,Rs)
    G_ans_SS.append(G)
    eta_ans_SS.append(eta)

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
ax[0,0].plot(matRs,np.abs(G_ans_SS),"k-")
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[1,0].plot(matRs,eta_ans_SS,"k-")
ax[1,0].set_xscale('log')
plt.show()