from __future__ import annotations
import numpy as np

# Parameter order is the operative 14-parameter Model16D indexing.
PARAM_NAMES = [
    "k1", "k_minus_1", "k2", "k_minus_2", "k3",
    "phi_x", "phi_v", "phi_l", "K", "Ks",
    "phi_s_minus_2", "phi_s3", "kd_ATP", "kd_Pi",
]

LOWER = np.array([
    0.5, 0.5, 0.5, 0.1, 0.5,
    0.5, 0.005, 1.0, 2000.0, 0.0,
    1.0, 1.0, 0.1, 0.05,
], dtype=float)

UPPER = np.array([
    200.0, 200.0, 200.0, 200.0, 200.0,
    10.0, 0.5, 5.0, 100000.0, 45.0,
    1000.0, 1000.0, 5.0, 5.0,
], dtype=float)

NONLINEAR_IDX = np.array([i for i in range(14) if i not in (8, 9)], dtype=int)
LOWER_NL = LOWER[NONLINEAR_IDX]
UPPER_NL = UPPER[NONLINEAR_IDX]

# Frozen rat Model16D configuration used in P1:
# md = 4: rapid-equilibrium ATP and Pi dependence
# sd = [0,0,0,1,1]: strain dependence on k_-2 and k3
MD = 4
SD = np.array([0, 0, 0, 1, 1], dtype=float)

LR95_DF1 = 3.841458820694124
N_FREQ = 17

CONDITIONS = {
    "baseline": (5.0, 1.0),
    "ATP0.1": (0.1, 1.0),
    "ATP1": (1.0, 1.0),
    "Pi0": (5.0, 1e-6),
    "Pi5": (5.0, 5.0),
}
