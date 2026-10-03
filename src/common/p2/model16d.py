from __future__ import annotations
import numpy as np

from .constants import N_FREQ

def _rates_and_states(params: np.ndarray, ATP: float, Pi: float):
    p = np.asarray(params, dtype=float)
    k1, km1, k2, km2, k3, phi_x, phi_v, phi_l, K, Ks, phi_s_m2, phi_s3, kd_ATP, kd_Pi = p

    ATP = float(ATP)
    Pi = max(float(Pi), 1e-12)

    # Frozen md=4: rapid-equilibrium metabolite dependence.
    km1e = km1 * Pi / (Pi + kd_Pi)
    k2e = k2 * kd_Pi / (Pi + kd_Pi)
    k3e = k3 * ATP / (ATP + kd_ATP)
    km2e = km2 * kd_ATP / (ATP + kd_ATP)

    Lmax = 2.3
    L0 = 2.2
    Z0 = 1.0 - (Lmax - L0) * phi_l / Lmax

    # Thermodynamic reverse rate k_-3, matching frozen MATLAB source.
    G_ATP0 = -30.0
    R = 8.314 / 1000.0
    T = 295.0
    ADP = 36e-6
    G_ATP = G_ATP0 + R*T*np.log((ADP*(Pi/1000.0))/(ATP/1000.0))
    km3 = k1*k2e*k3e / (km1e*km2e*np.exp(-G_ATP/(R*T)))

    sum_rates = k3e*(k1+k2e+km1e) + k1*k2e + km2e*(k1+km1e)
    B0 = k1*(km2e+k3e)/sum_rates * Z0
    C0 = k1*k2e/sum_rates * Z0
    A0 = Z0 - B0 - C0

    return {
        "k1":k1, "km1":km1e, "k2":k2e, "km2":km2e, "k3":k3e, "km3":km3,
        "phi_x":phi_x, "phi_v":phi_v, "phi_l":phi_l,
        "K":K, "Ks":Ks, "phi_s_m2":phi_s_m2, "phi_s3":phi_s3,
        "A0":A0, "B0":B0, "C0":C0, "Z0":Z0,
    }

def predict(params: np.ndarray, ATP: float, Pi: float, freqs: np.ndarray):
    """Return active complex modulus (MPa) and steady active stress (kPa)."""
    st = _rates_and_states(params, ATP, Pi)
    k1, km1, k2, km2, k3, km3 = st["k1"],st["km1"],st["k2"],st["km2"],st["k3"],st["km3"]
    phi_x,phi_v,phi_l = st["phi_x"],st["phi_v"],st["phi_l"]
    K,Ks = st["K"],st["Ks"]
    ps_m2,ps3 = st["phi_s_m2"],st["phi_s3"]
    A0,B0,C0 = st["A0"],st["B0"],st["C0"]

    Lmax=2.3; L0=2.2; xC0=0.01
    om=np.asarray(freqs,dtype=float)*2*np.pi
    omi=1j*om

    # sd=[0,0,0,1,1]: only k_-2 and k3 strain-dependent.
    d11=-(k1+km1+k2)
    d21=km2-k1
    d31=0.0
    d41=ps_m2*km2*C0
    du1=k1*phi_l/Lmax

    d12=k2-km3
    d22=-(k3+km2+km3)
    d32=0.0
    d42=-ps_m2*km2*C0 - ps3*k3*C0
    du2=km3*phi_l/Lmax

    d33=-phi_x/B0*(A0*k1+C0*km2)
    d44=-phi_x/C0*(B0*k2+A0*km3)

    HxB=phi_v*omi/(omi-d33)
    HxC=phi_v*omi/(omi-d44)
    den=(omi-d22)*(omi-d11)-d12*d21
    HC_strain=(d12*(d31*HxB+d41*HxC)+(d32*HxB+d42*HxC)*(omi-d11))/den
    HC_len=(du1*d12+du2*(omi-d11))/den
    HC=HC_strain+HC_len

    scale=L0/1000.0
    Y=scale*K*(B0*HxB+C0*HxC+HC*xC0)+scale*Ks
    stress=K*C0*xC0+Ks*0.3
    return Y, float(stress)

def steady_occupancy_c(params: np.ndarray, ATP: float, Pi: float) -> float:
    return float(_rates_and_states(params, ATP, Pi)["C0"])

def q_delta_stress(params: np.ndarray, target_met: tuple[float,float],
                   baseline_met: tuple[float,float]=(5.0,1.0)) -> float:
    _, fb = predict(params, *baseline_met, np.array([1.0]))
    _, ft = predict(params, *target_met, np.array([1.0]))
    return float(100.0*(ft/fb-1.0))

def nonlinear_basis(nl_params: np.ndarray, ATP: float, Pi: float, freqs: np.ndarray,
                    nonlinear_idx: np.ndarray):
    """
    Return quantities that make predictions linear in K and Ks:
       CM = K*yK + Ks*yKs
       F  = K*aK + Ks*0.3
    nl_params contains the 12 parameters excluding K and Ks.
    """
    p=np.empty(14,dtype=float)
    p[nonlinear_idx]=nl_params
    p[8]=1.0
    p[9]=0.0
    yK, fK = predict(p, ATP, Pi, freqs)
    # With K=1,Ks=0, fK=C0*xC0.
    yKs = np.full(len(freqs), 2.2/1000.0, dtype=complex)
    return yK, yKs, float(fK)
