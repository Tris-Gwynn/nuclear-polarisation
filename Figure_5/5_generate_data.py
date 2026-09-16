"""
run_figure5.py

Generates data for Figure 5: does resonance switching survive when the
polarised nuclear environment is enlarged from one nucleus to two, and
does the answer depend on whether both nuclei couple to the same
electron or one couples to each radical?

Case A: one_nucleus     -- one nucleus on donor, A1 = diag(1,-0.25,-0.5) mT
Case B: two_on_donor    -- two nuclei, both on donor, both tensor A1
                           (H_hf = S_D . A . (I1 + I2): total nuclear
                           spin conserved, I=0 sector decouples exactly)
Case C: donor_acceptor  -- one nucleus on donor, one on acceptor, both
                           tensor A1 (H_hf = S_D.A.I_D + S_A.A.I_A: no
                           common electron operator multiplies a
                           combined nuclear spin, so there is no
                           analogous I_tot conservation and no singlet
                           decoupling)

All three prepared with the same common Pz = +1, 0, -1 across whichever
nuclei are present. B parallel z throughout.

Note on the parity argument: U_z(pi) = exp(-i*pi*F_z), F_z = sum of all
individual electron + nuclear z-operators, commutes with H in all three
cases (each hyperfine tensor is diagonal in the lab frame, and B || z),
regardless of which electron each nucleus couples to. So Pz=+1 and
Pz=-1 landing in the same parity sector, and thus sharing exact
selection-rule exclusion from any resonance in the opposite sector,
should hold for Case C as well as Case B. What is NOT expected to
carry over to Case C is the specific nuclear-singlet decoupling
mechanism, since that relied on both nuclei acting on the same S_D.

Note on the midpoint identity: rho_n(P) is quadratic in the common P
for both two-nucleus cases (B and C), since it is a product of two
mixed single-nucleus states with the same P -- this is a fact about
the state parametrisation, not about which Hamiltonian acts on it. So
Phi_S(0) = [Phi_S(+1)+Phi_S(-1)]/2 is expected to hold exactly only
for Case A, not for B or C.

One task per (case, preparation) curve: 9 tasks total.
"""
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
# Fixed physical parameters (must match Figures 1-4)
# ----------------------------------------------------------------------
A1 = np.diag([1.0, -0.25, -0.5])   # mT, shared by all nuclei in every case

D_TENSOR = None
J_EX = 0.0

K_S = 1.0                 # PLACEHOLDER: must match Figures 1-4
K_T = 1.0                 # PLACEHOLDER: must match Figures 1-4
K_R = 0.0

USE_CISS = False
CHI_PERCENT = 0.0

THETA_FIELD, PHI_FIELD = 0.0, 0.0   # B parallel z

# ----------------------------------------------------------------------
# B0 sweep (PLACEHOLDER: must match Figures 1-4)
# ----------------------------------------------------------------------
B0_MIN = 0.0
B0_MAX = 1.0
N_B0 = 5000

OUTPUT_FILE = "Figure_5/figure5_data.npz"

N_WORKERS = 9   # one curve per worker: 3 cases x 3 preparations

# ----------------------------------------------------------------------
CASES = {
    "one_nucleus":    {"d_spins": [0.5],  "a_spins": [],     "a_tensor_d": [A1],      "a_tensor_a": []},
    "two_on_donor":   {"d_spins": [0.5, 0.5], "a_spins": [], "a_tensor_d": [A1, A1],  "a_tensor_a": []},
    "donor_acceptor": {"d_spins": [0.5],  "a_spins": [0.5],  "a_tensor_d": [A1],      "a_tensor_a": [A1]},
}

# Common Pz preparations applied to every nucleus present in the case
PREPARATIONS = {
    "Pz_plus":  +1.0,
    "P0":        0.0,
    "Pz_minus": -1.0,
}

# ----------------------------------------------------------------------
_worker_builder = None


def _init_worker(case_key):
    global _worker_builder
    case = CASES[case_key]
    _worker_builder = RPMBuilder(
        d_spins=case["d_spins"], a_spins=case["a_spins"],
        a_tensor_d=case["a_tensor_d"], a_tensor_a=case["a_tensor_a"],
        d_tensor=D_TENSOR, j_ex=J_EX,
        k_s=K_S, k_t=K_T, k_r=K_R,
        use_ciss=USE_CISS, chi_percent=CHI_PERCENT
    )


def _compute_curve(task):
    """Runs the full B0 sweep for one (case, preparation) combination.
    Uses initial_state, which applies the same p_val to every donor
    AND acceptor nucleus present -- exactly the common-P preparation
    needed for all three cases, including donor_acceptor."""
    case_key, prep_label, p_val, B0_values = task
    curve = np.zeros(len(B0_values))
    for i, B0 in enumerate(B0_values):
        S_sh, Tp_sh, T0_sh, Tm_sh = _worker_builder.yield_(
            B0, p_val=p_val, pol_axis='z', theta=THETA_FIELD, phi=PHI_FIELD
        )
        curve[i] = S_sh
    return case_key, prep_label, curve


if __name__ == "__main__":
    B0_values = np.linspace(B0_MIN, B0_MAX, N_B0)

    results = {}

    for case_key in CASES:
        tasks = [
            (case_key, prep_label, p_val, B0_values)
            for prep_label, p_val in PREPARATIONS.items()
        ]
        with ProcessPoolExecutor(
            max_workers=min(N_WORKERS, len(tasks)),
            initializer=_init_worker,
            initargs=(case_key,),
        ) as executor:
            for _, prep_label, curve in tqdm(
                executor.map(_compute_curve, tasks),
                total=len(tasks),
                desc=case_key,
            ):
                results[f"{case_key}_{prep_label}"] = curve

    np.savez(OUTPUT_FILE, B0_values=B0_values, **results)
    print(f"Saved to {OUTPUT_FILE}")

    # Diagnostic: midpoint deviation. Expected near-zero for one_nucleus
    # (exact affine result). NOT expected to vanish for two_on_donor or
    # donor_acceptor, since rho_n(P) is quadratic in the common P for
    # both two-nucleus cases -- a nonzero value here is the quadratic
    # term surviving, not an error. See module docstring.
    for case_key in CASES:
        p0 = results[f"{case_key}_P0"]
        pz_plus = results[f"{case_key}_Pz_plus"]
        pz_minus = results[f"{case_key}_Pz_minus"]
        dev = np.abs(p0 - 0.5 * (pz_plus + pz_minus)).max()
        print(f"{case_key}: max midpoint deviation = {dev:.3e}")