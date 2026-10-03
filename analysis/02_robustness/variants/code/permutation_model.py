"""Python translation of published XBModel_2024_linear_perms equations.

This module changes only the strain-dependence indicator. Metabolite mode is
fixed at published md=4 (rapid-equilibrium ATP and Pi) for the N comparison.
"""
from __future__ import annotations
import numpy as np

def active_strain_values(p, sd):
    idx=np.flatnonzero(np.asarray(sd,int))
    if len(idx)==0:return np.zeros(5,float)
    if len(idx)==1:
        v=np.zeros(5,float);v[idx[0]]=float(p[10]);return v
    v=np.zeros(5,float);v[idx[0]]=float(p[10]);v[idx[1]]=float(p[11]);return v

def predict(p,ATP,Pi,freqs,sd):
    p=np.asarray(p,float); ATP=float(ATP);Pi=max(float(Pi),1e-12)
    k1,km1,k2,km2,k3,phi_x,phi_v,phi_l,K,Ks=p[:10]
    kd_ATP,kd_Pi=p[12],p[13]
    km1=km1*Pi/(Pi+kd_Pi);k2=k2*kd_Pi/(Pi+kd_Pi)
    k3=k3*ATP/(ATP+kd_ATP);km2=km2*kd_ATP/(ATP+kd_ATP)
    G0=-30.0;R=8.314/1000.0;T=295.0;ADP=36e-6
    G=G0+R*T*np.log((ADP*(Pi/1000.0))/(ATP/1000.0))
    km3=k1*k2*k3/(km1*km2*np.exp(-G/(R*T)))
    Lmax=2.3;L0=2.2;xC0=0.01
    Z0=1-(Lmax-L0)*phi_l/Lmax
    sum_rates=k3*(k1+k2+km1)+k1*k2+km2*(k1+km1)
    B0=k1*(km2+k3)/sum_rates*Z0;C0=k1*k2/sum_rates*Z0;A0=Z0-B0-C0
    sv=active_strain_values(p,sd)
    d11=-(k1+km1+k2);d21=km2-k1
    d31=sv[0]*(-k1*A0)+sv[1]*(-km1*B0)+sv[2]*(k2*B0)
    d41=sv[3]*km2*C0;du1=k1*phi_l/Lmax
    d12=k2-km3;d22=-(k3+km2+km3)
    d32=sv[2]*(-k2*B0);d42=sv[3]*(-km2*C0)+sv[4]*(-k3*C0)
    du2=km3*phi_l/Lmax
    d33=-phi_x/B0*(A0*k1+C0*km2);du3=phi_v
    d44=-phi_x/C0*(B0*k2+A0*km3);du4=phi_v
    om=2*np.pi*np.asarray(freqs,float); omi=1j*om
    HxB=du3*omi/(omi-d33);HxC=du4*omi/(omi-d44)
    den=(omi-d22)*(omi-d11)-d12*d21
    HC_strain=(d12*(d31*HxB+d41*HxC)+(d32*HxB+d42*HxC)*(omi-d11))/den
    HC_len=(du1*d12+du2*(omi-d11))/den
    HC=HC_strain+HC_len
    scale=L0/1000.0
    Y=scale*K*(B0*HxB+C0*HxC+HC*xC0)+scale*Ks
    stress=K*C0*xC0+Ks*0.3
    return Y,float(stress)

def bases(nl,p_template,ATP,Pi,freqs,sd,nonlinear_idx):
    p=np.asarray(p_template,float).copy();p[np.asarray(nonlinear_idx,int)]=np.asarray(nl,float)
    p[8]=1.0;p[9]=0.0
    yk,fk=predict(p,ATP,Pi,freqs,sd)
    yks=np.full(len(freqs),2.2/1000.0,dtype=complex)
    return yk,yks,fk
