import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")
import numpy as np
import csv
from RPM_Experiment_Builder import RPMBuilder

D_SPINS = [0.5]
A_SPINS = []
A_TENSOR_D = [np.diag([1.0, -0.25, -0.5])]
A_TENSOR_A = []
D_TENSOR = None
J_EX = 0.0
K_S = 1.0  
K_T = 1.0  
K_R = 0.0
THETA, PHI = 0.0, 0.0

FIELDS = [0.1292, 0.33927, 0.35107, 0.3515, 0.3870]
NPZ = "Figure_3/figure3_main_data.npz"
OUT = "Figure_3/figure3_full_decomposition.csv"
TOP = 5

assert np.isclose(K_S, K_T)
k = K_S

builder = RPMBuilder(d_spins=D_SPINS, a_spins=A_SPINS, a_tensor_d=A_TENSOR_D,
                     a_tensor_a=A_TENSOR_A, d_tensor=D_TENSOR, j_ex=J_EX,
                     k_s=K_S, k_t=K_T, k_r=K_R, use_ciss=False, chi_percent=0.0)

P_S = builder.pop_ops['S'].full()
active_op = (builder.pop_ops['S'] + builder.pop_ops['Tp']
             + builder.pop_ops['T0'] + builder.pop_ops['Tm']).full()

data = np.load(NPZ)
Bgrid = data['B0_values']
curves = {1.0: data['curve_plus'], 0.0: data['curve_zero'], -1.0: data['curve_minus']}


def decompose(B0):
    H = builder._build_hamiltonian(B0, THETA, PHI)
    E, kets = H.eigenstates()
    U = np.column_stack([ket.full().ravel() for ket in kets])
    S_E = U.conj().T @ P_S @ U
    act_w = np.real(np.diag(U.conj().T @ active_op @ U))
    n = len(E)
    res = {}
    for p in (1.0, 0.0, -1.0):
        rho = builder.initial_state(p_val=p, pol_axis='z').full()
        rho_E = U.conj().T @ rho @ U
        diag = np.real(np.diag(rho_E) * np.diag(S_E))
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                C[i, j] = 2 * np.real(k * rho_E[i, j] * S_E[j, i] / (k + 1j * (E[i] - E[j])))
        res[p] = (diag, C)
    return E, act_w, res


rows = []
for B0 in FIELDS:
    E, act_w, res = decompose(B0)
    print(f"\nB0 = {B0} mT")
    for p in (1.0, 0.0, -1.0):
        diag, C = res[p]
        total = diag.sum() + C.sum()
        phi_main = np.interp(B0, Bgrid, curves[p])
        print(f"  Pz={p:+.0f}: diag={diag.sum():.5f} pairs={C.sum():.5f} "
              f"total={total:.5f} main={phi_main:.5f} diff={total - phi_main:.2e}")
    dP = res[1.0][0] - res[0.0][0]
    dC = res[1.0][1] - res[0.0][1]
    print(f"  Delta (Pz=+1 minus 0): diag total={dP.sum():.5f} pair total={dC.sum():.5f}")
    flat = [(abs(dC[i, j]), i, j) for i in range(len(E)) for j in range(i + 1, len(E))]
    for _, i, j in sorted(flat, reverse=True)[:TOP]:
        gap = E[i] - E[j]
        print(f"  pair ({i},{j}) 0-based, gap={gap:.3e}, gap/k={gap / k:.3e}, "
              f"dC={dC[i, j]:+.5f}, active weights {act_w[i]:.2f},{act_w[j]:.2f}")
        rows.append([B0, "pair", i, j, gap, gap / k, dC[i, j]])
    for i in np.argsort(-np.abs(dP))[:TOP]:
        print(f"  state {i} 0-based, ddiag={dP[i]:+.5f}, active weight {act_w[i]:.2f}")
        rows.append([B0, "diag", i, "", "", "", dP[i]])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["B0_mT", "kind", "i", "j", "gap", "gap_over_k", "delta_vs_Pz0_for_Pz+1"])
    w.writerows(rows)

    TABLE_PAIRS = [(0.1292, 13, 14), (0.3515, 1, 2), (0.3515, 11, 12), (0.3870, 11, 12)]
for B0, i, j in TABLE_PAIRS:
    E, _, res = decompose(B0)
    gap = E[i] - E[j]
    cs = [res[p][1][i, j] for p in (1.0, 0.0, -1.0)]
    print(B0, (i + 1, j + 1), f"|dE|/k={abs(gap) / k:.3e}", ["%.4f" % c for c in cs])

print(builder._build_hamiltonian(0.3515, THETA, PHI).dims)
E, act_w, _ = decompose(0.3515)
print("number of eigenstates:", len(E))
for n in range(len(E)):
    print(n + 1, f"E={E[n]:.4f}", f"active weight={act_w[n]:.3f}")

for B0, i, j in [(0.1292, 13, 14), (0.3515, 1, 2), (0.3515, 11, 12), (0.3870, 11, 12)]:
    E, act_w, _ = decompose(B0)
    active = list(np.where(act_w > 0.5)[0])
    print(B0, "active states:", len(active), "pair:", (active.index(i) + 1, active.index(j) + 1))