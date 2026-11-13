import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from RPM_Experiment_Builder import RPMBuilder

# ----------------------------------------------------------------------
# Fixed physical parameters (one-nucleus setup, matches Figures 1-4)
# ----------------------------------------------------------------------
A1 = np.diag([1.0, -0.25, -0.5])
K = 1.0                      # k_S = k_T = K (closed form requires equal rates)

# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------
B0_MIN, B0_MAX, N_B0 = 0.0, 1.0, 500
N_THETA = 361                 # 5 degree steps over [0, pi], includes pi/4 and both poles
N_PHI = 720                   # 5 degree steps over [0, 2pi), endpoint excluded (phi = 2pi is phi = 0)

OUTPUT_FILE = "Figure_8/figure8_orientation_map.npz"

N_WORKERS = 28
CHUNKSIZE = 16

AXES = ("x", "y", "z")

# ----------------------------------------------------------------------
_W = None   # per-worker cache


def _make_builder():
    return RPMBuilder(
        d_spins=[0.5], a_spins=[],
        a_tensor_d=[A1], a_tensor_a=[],
        d_tensor=None, j_ex=0.0,
        k_s=K, k_t=K, k_r=0.0,
        use_ciss=False, chi_percent=0.0,
    )


def _prepare():
    """Build fixed matrices: H(B) = H0 + B0 * (n . Z), singlet projector, Delta rho."""
    b = _make_builder()
    H0 = b._build_hamiltonian(0.0).full()
    # Zeeman operator per unit field along each axis (Zeeman is linear in B)
    Zx = b._build_hamiltonian(1.0, np.pi / 2, 0.0).full() - H0
    Zy = b._build_hamiltonian(1.0, np.pi / 2, np.pi / 2).full() - H0
    Zz = b._build_hamiltonian(1.0, 0.0, 0.0).full() - H0
    P_S = b.pop_ops["S"].full()
    rho0 = b.initial_state(p_val=0.0).full()
    drho = {a: b.initial_state(p_val=1.0, pol_axis=a).full() - rho0 for a in AXES}
    return dict(b=b, H0=H0, Z=(Zx, Zy, Zz), P_S=P_S, drho=drho)


def _hamiltonian(W, B0, theta, phi):
    n = (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    return W["H0"] + B0 * (n[0] * W["Z"][0] + n[1] * W["Z"][1] + n[2] * W["Z"][2])


def _delta_phi(W, H):
    """Closed-form Delta Phi_alpha for alpha = x, y, z at Hamiltonian H."""
    E, U = np.linalg.eigh(H)
    Ud = U.conj().T
    S_E = Ud @ W["P_S"] @ U
    M = K / (K + 1j * (E[:, None] - E[None, :]))      # k / (k + i(E_i - E_j))
    out = []
    for a in AXES:
        r_E = Ud @ W["drho"][a] @ U
        out.append(np.real(np.sum(r_E * S_E.T * M)))   # sum_ij rho_ij S_ji M_ij
    return out


def _init_worker():
    global _W
    _W = _prepare()


def _compute_orientation(task):
    it, ip, theta, phi, B0_values = task
    curves = np.zeros((3, len(B0_values)))
    for i, B0 in enumerate(B0_values):
        curves[:, i] = _delta_phi(_W, _hamiltonian(_W, B0, theta, phi))
    return it, ip, curves


# ----------------------------------------------------------------------
def _validate():
    W = _prepare()
    b = W["b"]
    rng = np.random.default_rng(1)

    print("1. Hamiltonian assembly check")
    for _ in range(3):
        B0, th, ph = rng.uniform(0, 1), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        err = np.abs(_hamiltonian(W, B0, th, ph) - b._build_hamiltonian(B0, th, ph).full()).max()
        print(f"   B0={B0:.3f} theta={th:.3f} phi={ph:.3f}  max|dH| = {err:.2e}")

    print("2. Closed form vs mesolve (Delta Phi_alpha, theta=phi=pi/4)")
    for B0 in (0.14, 0.30):
        cf = _delta_phi(W, _hamiltonian(W, B0, np.pi / 4, np.pi / 4))
        y0 = b.yield_(B0, p_val=0.0, theta=np.pi / 4, phi=np.pi / 4)[0]
        for a, c in zip(AXES, cf):
            ym = b.yield_(B0, p_val=1.0, pol_axis=a, theta=np.pi / 4, phi=np.pi / 4)[0] - y0
            print(f"   B0={B0}  {a}: closed={c:+.5f}  mesolve={ym:+.5f}  ratio={ym / c if c else float('nan'):.4f}")

    print("3. D2 symmetry check (|Delta Phi| at n vs n folded into x,y >= 0)")
    for _ in range(4):
        B0 = rng.uniform(0.05, 0.5)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        th, ph = np.arccos(v[2]), np.arctan2(v[1], v[0])
        va = v.copy()
        if va[0] < 0 and va[1] < 0: va[[0, 1]] *= -1          # pi rotation about z
        elif va[0] < 0: va[[0, 2]] *= -1                     # pi rotation about y
        elif va[1] < 0: va[[1, 2]] *= -1                     # pi rotation about x
        tha, pha = np.arccos(va[2]), np.arctan2(va[1], va[0])
        d1 = np.abs(_delta_phi(W, _hamiltonian(W, B0, th, ph)))
        d2 = np.abs(_delta_phi(W, _hamiltonian(W, B0, tha, pha)))
        print(f"   B0={B0:.3f}  max diff = {np.abs(d1 - d2).max():.2e}  (values {np.round(d1, 4)})")
    print()


if __name__ == "__main__":
    _validate()

    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)
    thetas = np.linspace(0.0, np.pi, N_THETA)
    phis = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)

    tasks = [(it, ip, th, ph, B0_values)
             for it, th in enumerate(thetas)
             for ip, ph in enumerate(phis)
             if not (it in (0, N_THETA - 1) and ip > 0)]   # each pole is one direction, compute once

    curves = np.zeros((N_THETA, N_PHI, 3, N_B0), dtype=np.float32)

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        for it, ip, c in tqdm(ex.map(_compute_orientation, tasks, chunksize=CHUNKSIZE),
                              total=len(tasks)):
            curves[it, ip] = c

    curves[0, 1:] = curves[0, 0]                # fill the pole rows from the single pole runs
    curves[-1, 1:] = curves[-1, 0]

    abs_c = np.abs(curves.astype(np.float64))
    _trap = getattr(np, "trapezoid", None) or np.trapz
    area_abs = _trap(abs_c, B0_values, axis=-1)
    imax = np.argmax(abs_c, axis=-1)
    max_abs = np.take_along_axis(abs_c, imax[..., None], axis=-1)[..., 0]
    B_at_max = B0_values[imax]

    np.savez(OUTPUT_FILE, B0_values=B0_values, thetas=thetas, phis=phis, axes=np.array(AXES),
             curves=curves, area_abs=area_abs, max_abs=max_abs, B_at_max=B_at_max)
    print(f"Saved to {OUTPUT_FILE}")

    # Quick checks against known results
    iz = 0
    ipi4 = np.argmin(np.abs(thetas - np.pi / 4)), np.argmin(np.abs(phis - np.pi / 4))
    print("B || z (theta=0) max|dPhi| x,y,z:", np.round(max_abs[iz, 0], 4), "(expect 0, 0, ~0.163; infinite-time, so ~0.7% above mesolve)")
    print("theta=phi=pi/4  max|dPhi| x,y,z:", np.round(max_abs[ipi4], 4), "(expect ~0.053, 0.044, 0.072, slightly above)")