"""
run_tolerance_convergence.py

Solver-tolerance convergence check for the Numerical Reliability
section. Not a curve re-run: at four representative points, compares
the yield computed with production settings (BDF, default qutip
tolerances -- matching RPM_Experiment_Builder.py's current
options={'method': 'bdf', 'nsteps': 1000000}, no rtol/atol override)
against BDF with tightened tolerances (rtol=1e-9, atol=1e-11).

RPMBuilder.yield_ does not expose a way to override tolerances per
call, so this reimplements its logic directly using the builder's
public H/rho0/c_ops/pop_ops construction, exactly as the Figure 3
diagnostic scripts already do.

Four points:
  1. one_nucleus, Pz=+1, B0=0.129 mT   (baseline resonance)
  2. one_nucleus, Pz=+1, B0=0.387 mT   (baseline resonance)
  3. two_on_donor, P=0,  B0=0.331 mT   (two-nucleus resonance)
  4. slow_triplet (k_T=0.1), Pz=+1, B0=0.3868 mT (unequal-rate peak)
"""
import sys
sys.path.insert(0, "/home/tristengwynn/nuclear-polarisation/RPM_System")

import numpy as np
import qutip as qt
from RPM_Experiment_Builder import RPMBuilder

A1 = np.diag([1.0, -0.25, -0.5])
THETA, PHI = 0.0, 0.0   # B parallel z

PRODUCTION_OPTIONS = {'method': 'bdf', 'nsteps': 1000000}
TIGHT_OPTIONS = {'method': 'bdf', 'nsteps': 1000000, 'rtol': 1e-9, 'atol': 1e-11}

POINTS = {
    "one_nucleus_B0=0.129": {
        "builder_kwargs": dict(d_spins=[0.5], a_spins=[], a_tensor_d=[A1],
                                a_tensor_a=[], d_tensor=None, j_ex=0.0,
                                k_s=1.0, k_t=1.0, k_r=0.0,
                                use_ciss=False, chi_percent=0.0),
        "p_val": +1.0, "pol_axis": 'z', "B0": 0.129,
    },
    "one_nucleus_B0=0.387": {
        "builder_kwargs": dict(d_spins=[0.5], a_spins=[], a_tensor_d=[A1],
                                a_tensor_a=[], d_tensor=None, j_ex=0.0,
                                k_s=1.0, k_t=1.0, k_r=0.0,
                                use_ciss=False, chi_percent=0.0),
        "p_val": +1.0, "pol_axis": 'z', "B0": 0.387,
    },
    "two_on_donor_B0=0.331": {
        "builder_kwargs": dict(d_spins=[0.5, 0.5], a_spins=[], a_tensor_d=[A1, A1],
                                a_tensor_a=[], d_tensor=None, j_ex=0.0,
                                k_s=1.0, k_t=1.0, k_r=0.0,
                                use_ciss=False, chi_percent=0.0),
        "p_val": 0.0, "pol_axis": 'z', "B0": 0.331,
    },
    "slow_triplet_B0=0.3868": {
        "builder_kwargs": dict(d_spins=[0.5], a_spins=[], a_tensor_d=[A1],
                                a_tensor_a=[], d_tensor=None, j_ex=0.0,
                                k_s=1.0, k_t=0.1, k_r=0.0,
                                use_ciss=False, chi_percent=0.0),
        "p_val": +1.0, "pol_axis": 'z', "B0": 0.3868,
    },
}


def compute_yield(builder, B0, p_val, pol_axis, options):
    """Reimplements RPMBuilder.yield_ with a custom options dict."""
    t_max = 5 / min(builder.k_s, builder.k_t)
    H = builder._build_hamiltonian(B0, THETA, PHI)
    rho0 = builder.initial_state(p_val=p_val, pol_axis=pol_axis)
    keys = ('S_sh', 'Tp_sh', 'T0_sh', 'Tm_sh')
    e_ops = [builder.pop_ops[k] for k in keys]
    res = qt.mesolve(H, rho0, [0, t_max], builder.c_ops, e_ops=e_ops,
                      options=options)
    S_sh = float(np.real(res.expect[0][-1]))
    return S_sh


print(f"{'point':30s} {'production':>14s} {'tight':>14s} {'abs diff':>12s}")
for label, point in POINTS.items():
    builder = RPMBuilder(**point["builder_kwargs"])
    S_production = compute_yield(builder, point["B0"], point["p_val"],
                                  point["pol_axis"], PRODUCTION_OPTIONS)
    S_tight = compute_yield(builder, point["B0"], point["p_val"],
                             point["pol_axis"], TIGHT_OPTIONS)
    diff = abs(S_production - S_tight)
    print(f"{label:30s} {S_production:14.8f} {S_tight:14.8f} {diff:12.3e}")