"""
Q1: explicit dependence of P_S(t) on off-diagonal nuclear initial state terms.

Three panels showing the same three polarisation conditions (unpolarised, z-pol,
x-pol) under three symmetry regimes:
  Left   - isotropic HF, field on z       (axial symmetry intact, off-diagonal inert)
  Centre - isotropic HF, field tilted 45  (symmetry broken by field tilt)
  Right  - anisotropic HF, field on z     (symmetry broken by off-diagonal tensor elements)

Uses RPMBuilder throughout for consistency with all other Q figures.
System: 2 electrons + 1 spin-1/2 nucleus (acceptor only), coherent (k=0).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RPM_Experiment_Builder import RPMBuilder
from cases import iso, TENSORS, R_TRP_TO_FAD
from cases import TENSORS, R_TRP_TO_FAD

# ── parameters ────────────────────────────────────────────────────────────────
B0     = 0.1          # mT
A_ISO  = iso(1.0)     # isotropic 1 mT, conserves Jz for field on z
A_ANISO = R_TRP_TO_FAD @ TENSORS['TrpH']['H1'] @ R_TRP_TO_FAD.T  # off-diagonal elements break Jz

TIMES  = np.linspace(0, 0.05, 500)   # μs
TS_NS  = TIMES * 1e3                  # display in ns

# ── builders ──────────────────────────────────────────────────────────────────
b_iso   = RPMBuilder(d_spins=[], a_spins=[0.5],
                     a_tensor_d=[], a_tensor_a=[A_ISO],
                     k_s=0.0, k_t=0.0)

b_aniso = RPMBuilder(d_spins=[], a_spins=[0.5],
                     a_tensor_d=[], a_tensor_a=[A_ANISO],
                     k_s=0.0, k_t=0.0)

# ── compute traces (index 0 = active singlet S population) ────────────────────
def run(builder, theta=0.0):
    unpol = builder.simulate_dynamics(B0, TIMES, p_val=0.0, pol_axis='z', theta=theta)[0]
    zpol  = builder.simulate_dynamics(B0, TIMES, p_val=1.0, pol_axis='z', theta=theta)[0]
    xpol  = builder.simulate_dynamics(B0, TIMES, p_val=1.0, pol_axis='x', theta=theta)[0]
    return unpol, zpol, xpol

A_unpol, A_zpol, A_xpol   = run(b_iso,   theta=0.0)         # left panel
B_unpol, B_zpol, B_xpol   = run(b_iso,   theta=np.pi/4)     # centre panel
C_unpol, C_zpol, C_xpol   = run(b_aniso, theta=0.0)         # right panel

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

c_unpol, c_zpol, c_xpol = 'k', 'C3', 'C0'
lw = 1.8

panels = [
    (axes[0], A_unpol, A_zpol, A_xpol,
     'Isotropic HF, field on z\n(axial symmetry intact)',
     'x-pol $=$ unpol\n(off-diagonal inert)'),
    (axes[1], B_unpol, B_zpol, B_xpol,
     'Isotropic HF, field tilted 45$^\\circ$\n(symmetry broken by field tilt)',
     'x-pol $\\neq$ unpol\n(off-diagonal active)'),
    (axes[2], C_unpol, C_zpol, C_xpol,
     'Anisotropic HF, field on z\n(symmetry broken by tensor)',
     'x-pol $\\neq$ unpol\n(off-diagonal active)'),
]

for ax, unpol, zpol, xpol, title, note in panels:
    ax.plot(TS_NS, unpol, color=c_unpol, lw=lw,       label='unpolarised')
    ax.plot(TS_NS, zpol,  color=c_zpol,  lw=lw, ls=':', label='z-pol (diagonal)')
    ax.plot(TS_NS, xpol,  color=c_xpol,  lw=lw, ls='--', label='x-pol (off-diagonal)')
    ax.set_xlabel('time (ns)')
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, TS_NS[-1])
    ax.text(0.04, 0.06, note, transform=ax.transAxes,
            fontsize=8, color=c_xpol, va='bottom')

axes[0].set_ylabel(r'singlet probability $P_S(t)$')
axes[0].legend(frameon=False, fontsize=8, loc='upper right')

plt.suptitle(r'Q1: dependence of $P_S(t)$ on off-diagonal nuclear terms', fontsize=11)
plt.tight_layout()

fname = 'q1_offdiagonal_final.png'
plt.savefig(fname, dpi=200, bbox_inches='tight')
plt.close()
print('saved:', fname)