"""
Generalised RPM solver wrapper.
Same physics engine (core/solver/cases). Adds: case-or-custom construction,
isotropic hyperfine toggle, custom dipolar tensor, defaulted kinetics,
real-valued yields, and a coherent/open distinction via the kinetic rates.
"""
import numpy as np
import qutip as qt
from constants import GYRO_E, GYRO_H1, GYRO_N14
from core import (NSpinRPMSystem,
                  get_n_spin_anisotropic_hyperfine,
                  get_n_spin_zeeman,
                  get_n_spin_dipolar,
                  get_n_spin_exchange)
from solver import NPolarizedSolver
from cases import get_nuclear_case


def _isotropise(tensor_list):
    return [np.eye(3) * (np.trace(np.asarray(t, float)) / 3.0) for t in tensor_list]


class RPMBuilder:
    DEFAULT_D_TENSOR = np.array([
        [ 0.030687, -0.338269,  0.136643],
        [-0.338269, -0.218231,  0.195874],
        [ 0.136643,  0.195874,  0.187544],
    ])

    def __init__(self, active_case=None, *,
                 d_spins=None, a_spins=None,
                 a_tensor_d=None, a_tensor_a=None,
                 isotropic_hf=False,
                 use_dipolar=False, d_tensor=None,
                 j_ex=0.0,
                 use_ciss=False,
                 k_s=1.0, k_t=1.0, k_r=0.0, chi_percent=0.0,
                 case_kwargs=None):

        if active_case is not None:
            cfg = get_nuclear_case(active_case, **(case_kwargs or {}))
            if d_spins is None:    d_spins = cfg['D_SPINS']
            if a_spins is None:    a_spins = cfg['A_SPINS']
            if a_tensor_d is None: a_tensor_d = cfg['A_TENSOR_D_LIST']
            if a_tensor_a is None: a_tensor_a = cfg['A_TENSOR_A_LIST']
        else:
            missing = [n for n, v in (('d_spins', d_spins), ('a_spins', a_spins),
                                      ('a_tensor_d', a_tensor_d), ('a_tensor_a', a_tensor_a))
                       if v is None]
            if missing:
                raise ValueError(f"Custom build needs {missing} when active_case is None.")

        self.d_spins = list(d_spins)
        self.a_spins = list(a_spins)
        a_tensor_d = [np.asarray(t, float) for t in a_tensor_d]
        a_tensor_a = [np.asarray(t, float) for t in a_tensor_a]

        self.isotropic_hf = isotropic_hf
        if isotropic_hf:
            a_tensor_d = _isotropise(a_tensor_d)
            a_tensor_a = _isotropise(a_tensor_a)
        self.a_tensor_d = a_tensor_d
        self.a_tensor_a = a_tensor_a

        self.use_dipolar = use_dipolar
        if use_dipolar:
            self.d_tensor = (self.DEFAULT_D_TENSOR.copy() if d_tensor is None
                             else np.asarray(d_tensor, float))
        else:
            self.d_tensor = np.zeros((3, 3))

        self.j_ex = j_ex
        self.k_s, self.k_t, self.k_r = k_s, k_t, k_r
        self.chi_init = chi_percent / 100.0 * np.pi / 2
        self.coherent = (k_s == 0.0 and k_t == 0.0 and k_r == 0.0)

        self.sys_rpm = NSpinRPMSystem(d_spins=self.d_spins, a_spins=self.a_spins,
                                      use_ciss=use_ciss)
        self.solver = NPolarizedSolver(self.sys_rpm)
        self.c_ops = self.solver.get_collapse_ops(self.k_s, self.k_t,
                                                  kr=self.k_r, chi=self.chi_init)
        self.pop_ops = self.solver.get_population_ops()

    def _get_hamiltonian(self, B0, theta=0.0, phi=0.0,
                         b1_rf_mt=0.0, rf_freq_mhz=0.0, rf_ax='z', fn='sin(w * t)'):
        H_hf = get_n_spin_anisotropic_hyperfine(self.sys_rpm, self.a_tensor_d,
                                                self.a_tensor_a, theta=theta, phi=phi)
        H_z = get_n_spin_zeeman(self.sys_rpm, B0, theta=theta, phi=phi)
        H_dip = get_n_spin_dipolar(self.sys_rpm, self.d_tensor, theta=theta, phi=phi)
        H_ex = get_n_spin_exchange(self.sys_rpm, self.j_ex)
        H_0 = H_hf + H_z + H_dip + H_ex
        args = {'w': rf_freq_mhz * 2 * np.pi}

        if b1_rf_mt != 0.0:
            H_1 = (b1_rf_mt * GYRO_E) * (self.sys_rpm.SD[rf_ax] + self.sys_rpm.SA[rf_ax])
            for sv, nuc in zip(self.sys_rpm.d_spins, self.sys_rpm.ID):
                H_1 += (b1_rf_mt * (GYRO_H1 if sv == 0.5 else GYRO_N14)) * nuc[rf_ax]
            for sv, nuc in zip(self.sys_rpm.a_spins, self.sys_rpm.IA):
                H_1 += (b1_rf_mt * (GYRO_H1 if sv == 0.5 else GYRO_N14)) * nuc[rf_ax]
            return [H_0, [H_1, fn]], args
        return H_0, args

    def _get_initial_state(self, p_val, axis='z'):
        axis_map = {'x': [p_val, 0.0, 0.0], 'y': [0.0, p_val, 0.0], 'z': [0.0, 0.0, p_val]}
        p_vec = axis_map.get(axis, [0.0, 0.0, p_val])
        p_d = [p_vec for _ in self.d_spins]
        p_a = [p_vec for _ in self.a_spins]
        return self.solver.get_initial_rho(p_d, p_a, chi_init=self.chi_init)

    def simulate_yield(self, B0, t_max=5.0, p_val=0.0, pol_axis='z',
                       theta=0.0, phi=0.0,
                       b1_rf_mt=0.0, rf_freq_mhz=0.0, rf_ax='z', fn='sin(w * t)'):
        """Asymptotic shelved yields (S, Tp, T0, Tm), real-valued."""
        H, args = self._get_hamiltonian(B0, theta, phi, b1_rf_mt, rf_freq_mhz, rf_ax, fn)
        rho0 = self._get_initial_state(p_val, pol_axis)
        e_ops = [self.pop_ops[k] for k in ('S_sh', 'Tp_sh', 'T0_sh', 'Tm_sh')]
        res = qt.mesolve(H, rho0, [0, t_max], self.c_ops, e_ops=e_ops,
                         args=args, options={'nsteps': 100000})
        return tuple(float(np.real(ex[-1])) for ex in res.expect)

    def simulate_dynamics(self, B0, times_array, p_val=0.0, pol_axis='z',
                          theta=0.0, phi=0.0,
                          b1_rf_mt=0.0, rf_freq_mhz=0.0, rf_ax='z', fn='sin(w * t)'):
        """Full traces: active (S, Tp, T0, Tm) then shelved (S_sh, Tp_sh, T0_sh, Tm_sh)."""
        H, args = self._get_hamiltonian(B0, theta, phi, b1_rf_mt, rf_freq_mhz, rf_ax, fn)
        rho0 = self._get_initial_state(p_val, pol_axis)
        keys = ('S', 'Tp', 'T0', 'Tm', 'S_sh', 'Tp_sh', 'T0_sh', 'Tm_sh')
        e_ops = [self.pop_ops[k] for k in keys]
        res = qt.mesolve(H, rho0, times_array, self.c_ops, e_ops=e_ops,
                         args=args, options={'nsteps': 100000})
        return tuple(np.real(ex) for ex in res.expect)