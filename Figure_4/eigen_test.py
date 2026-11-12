import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")
import numpy as np
from RPM_Experiment_Builder import RPMBuilder

A = np.diag([1.0, -0.25, -0.5])
K = 1.0
BS = np.linspace(0.001, 1.0, 5000)
GEOMS = {"B along z": (0.0, 0.0), "off-axis": (np.pi / 4, np.pi / 4)}

builder = RPMBuilder(d_spins=[0.5], a_spins=[], a_tensor_d=[A], a_tensor_a=[],
                     d_tensor=None, j_ex=0.0, k_s=K, k_t=K, k_r=0.0,
                     use_ciss=False, chi_percent=0.0)
P_S = builder.pop_ops['S'].full()
rho0 = builder.initial_state(p_val=0.0, pol_axis='z').full()
rhoP = {a: builder.initial_state(p_val=1.0, pol_axis=a).full() for a in "xyz"}


def split(B0, theta, phi):
    H = builder._build_hamiltonian(B0, theta, phi)
    E, kets = H.eigenstates()
    U = np.column_stack([ket.full().ravel() for ket in kets])
    S_E = U.conj().T @ P_S @ U
    dE = E[:, None] - E[None, :]

    def parts(rho):
        r = U.conj().T @ rho @ U
        diag = np.real(np.diag(r) * np.diag(S_E))
        C = np.triu(2 * np.real(K * r * S_E.T / (K + 1j * dE)), 1)
        return diag, C

    d0, C0 = parts(rho0)
    res = {}
    for a in "xyz":
        d, C = parts(rhoP[a])
        res[a] = (d - d0, C - C0)
    return res


for name, (th, ph) in GEOMS.items():
    mx_pair = {a: 0.0 for a in "xyz"}
    mx_diag = {a: 0.0 for a in "xyz"}
    mx_tot = {a: 0.0 for a in "xyz"}
    for B0 in BS:
        res = split(B0, th, ph)
        for a in "xyz":
            dd, dC = res[a]
            mx_pair[a] = max(mx_pair[a], np.abs(dC).max())
            mx_diag[a] = max(mx_diag[a], np.abs(dd).max())
            mx_tot[a] = max(mx_tot[a], abs(dd.sum() + dC.sum()))
    print(name)
    for a in "xyz":
        print(f"  P_{a}: max|dPhi|={mx_tot[a]:.5f}  max single-pair |dC|={mx_pair[a]:.5f}  max single-state |d diag|={mx_diag[a]:.5f}")