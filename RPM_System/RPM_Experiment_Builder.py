import numpy as np
import qutip as qt
from core import (NSpinRPMSystem, get_n_spin_anisotropic_hyperfine,
                  get_n_spin_zeeman, get_n_spin_dipolar, get_n_spin_exchange)
from solver import NPolarizedSolver


class RPMBuilder:
    def __init__(self, *,
                 d_spins, a_spins,
                 a_tensor_d, a_tensor_a,
                 d_tensor=None,
                 j_ex=0.0,
                 k_s=1.0, k_t=1.0, k_r=0.0,
                 use_ciss=False, chi_percent=0.0):
        """
        Parameters
        ----------
        d_spins      : list of spin quantum numbers for donor nuclei, e.g. [1.0, 0.5]
        a_spins      : list of spin quantum numbers for acceptor nuclei
        a_tensor_d   : list of 3x3 hyperfine tensors (mT) for donor nuclei
        a_tensor_a   : list of 3x3 hyperfine tensors (mT) for acceptor nuclei
        d_tensor     : 3x3 dipolar tensor (mT). None = no dipolar coupling.
        j_ex         : exchange coupling scalar (mT)
        k_s, k_t     : singlet/triplet reaction rates (MHz)
        k_r          : CISS backscatter rate (MHz)
        chi_percent  : CISS chiral angle as percent of pi/2
        """
        self.d_spins = list(d_spins)
        self.a_spins = list(a_spins)
        self.a_tensor_d = [np.asarray(t, float) for t in a_tensor_d]
        self.a_tensor_a = [np.asarray(t, float) for t in a_tensor_a]
        self.d_tensor   = np.asarray(d_tensor, float) if d_tensor is not None else np.zeros((3, 3))
        self.j_ex       = float(j_ex)
        self.k_s, self.k_t, self.k_r = k_s, k_t, k_r
        self.chi_init   = chi_percent / 100.0 * np.pi / 2

        self.sys_rpm = NSpinRPMSystem(d_spins=self.d_spins, a_spins=self.a_spins,
                                      use_ciss=use_ciss)
        self.solver  = NPolarizedSolver(self.sys_rpm)
        self.c_ops   = self.solver.get_collapse_ops(k_s, k_t, kr=k_r, chi=self.chi_init)
        self.pop_ops = self.solver.get_population_ops()

    # ------------------------------------------------------------------
    # Hamiltonian
    # ------------------------------------------------------------------
    def _build_hamiltonian(self, B0, theta=0.0, phi=0.0, theta_hf=0.0, phi_hf=0.0, theta_d=0.0, phi_d=0.0):
        H_hf = get_n_spin_anisotropic_hyperfine(self.sys_rpm, self.a_tensor_d, self.a_tensor_a, 
                                                theta=theta_hf, phi=phi_hf)
        H_z   = get_n_spin_zeeman(self.sys_rpm, B0, theta=theta, phi=phi)
        H_dip = get_n_spin_dipolar(self.sys_rpm, self.d_tensor, theta=theta_d, phi=phi_d)
        H_ex  = get_n_spin_exchange(self.sys_rpm, self.j_ex)
        return H_hf + H_z + H_dip + H_ex
               
        
    # ------------------------------------------------------------------
    # Initial states
    # ------------------------------------------------------------------
    def _polarisation_vec(self, p_val, axis):
        return {'x': [p_val, 0.0, 0.0],
                'y': [0.0, p_val, 0.0],
                'z': [0.0, 0.0, p_val]}.get(axis, [0.0, 0.0, p_val])

    def initial_state(self, p_val=0.0, pol_axis='z'):
        """Standard polarised/unpolarised initial state, same vector on all nuclei."""
        p_vec = self._polarisation_vec(p_val, pol_axis)
        p_d   = [p_vec for _ in self.d_spins]
        p_a   = [p_vec for _ in self.a_spins]
        return self.solver.get_initial_rho(p_d, p_a, chi_init=self.chi_init)

    def initial_state_custom(self, p_d, p_a):
        """Per-nucleus polarisation vectors. p_d and p_a are lists of [px,py,pz]."""
        return self.solver.get_initial_rho(p_d, p_a, chi_init=self.chi_init)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def dynamics(self, B0, times, p_val=0.0, pol_axis='z',
                 theta=0.0, phi=0.0, rho0=None):
        H    = self._build_hamiltonian(B0, theta, phi)
        rho0 = rho0 if rho0 is not None else self.initial_state(p_val, pol_axis)
        keys = ('S', 'Tp', 'T0', 'Tm', 'S_sh', 'Tp_sh', 'T0_sh', 'Tm_sh')
        e_ops = [self.pop_ops[k] for k in keys]
        res  = qt.mesolve(H, rho0, times, self.c_ops, e_ops=e_ops,
                          options={'method': 'bdf','nsteps': 1000000})
        return tuple(np.real(ex) for ex in res.expect)

    def yield_(self, B0, p_val=0.0, pol_axis='z',
               theta=0.0, phi=0.0, rho0=None):
        t_max = 5 / min(self.k_s, self.k_t)
        H    = self._build_hamiltonian(B0, theta, phi)
        rho0 = rho0 if rho0 is not None else self.initial_state(p_val, pol_axis)
        keys = ('S_sh', 'Tp_sh', 'T0_sh', 'Tm_sh')
        e_ops = [self.pop_ops[k] for k in keys]
        res  = qt.mesolve(H, rho0, [0, t_max], self.c_ops, e_ops=e_ops,
                          options={'method': 'bdf','nsteps': 1000000})
        return tuple(float(np.real(ex[-1])) for ex in res.expect) #type: ignore