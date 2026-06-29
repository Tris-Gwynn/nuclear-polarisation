"""
Centralized solver engine for RPM simulations. 
Encapsulates Hamiltonian assembly, state generation, and numerical integration.
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

class RPMBuilder:
    def __init__(self, active_case, 
                 use_dipolar=True, 
                 j_ex=0.0, 
                 use_ciss=False, 
                 k_s=1.0, 
                 k_t=1.0, 
                 k_r=0.0, 
                 chi_percent=0):
        
        self.sys_config = get_nuclear_case(active_case)
        self.d_spins = self.sys_config['D_SPINS']
        self.a_spins = self.sys_config['A_SPINS']
        self.a_tensor_d = self.sys_config['A_TENSOR_D_LIST']
        self.a_tensor_a = self.sys_config['A_TENSOR_A_LIST']
        
        self.j_ex = j_ex
        self.use_dipolar = use_dipolar
        if self.use_dipolar:
            self.d_tensor = np.array([
                [ 0.030687, -0.338269,  0.136643], 
                [-0.338269, -0.218231,  0.195874], 
                [ 0.136643,  0.195874,  0.187544]
            ])
        else:
            self.d_tensor = np.zeros((3, 3))
            
        self.k_s = k_s
        self.k_t = k_t
        self.k_r = k_r
        self.chi_init = chi_percent / 100.0 * np.pi / 2
        
        self.sys_rpm = NSpinRPMSystem(d_spins=self.d_spins, a_spins=self.a_spins, use_ciss=use_ciss)
        self.solver = NPolarizedSolver(self.sys_rpm)
        
        # Pre-build operators as they do not depend on external fields
        self.c_ops = self.solver.get_collapse_ops(self.k_s, self.k_t, kr=self.k_r, chi=self.chi_init)
        self.pop_ops = self.solver.get_population_ops()
        
    def _get_hamiltonian(self, 
                         B0, theta=0.0, phi=0.0, 
                         b1_rf_mt=0.0, rf_freq_mhz=0.0, rf_ax='z', fn='sin(w * t)'):
        """Internal method to construct the Hamiltonian."""
        H_hf = get_n_spin_anisotropic_hyperfine(self.sys_rpm, self.a_tensor_d, self.a_tensor_a, theta=theta, phi=phi)
        H_z = get_n_spin_zeeman(self.sys_rpm, B0, theta=theta, phi=phi)
        H_dip = get_n_spin_dipolar(self.sys_rpm, self.d_tensor, theta=theta, phi=phi)
        H_ex = get_n_spin_exchange(self.sys_rpm, self.j_ex)
        
        H_0 = H_hf + H_z + H_dip + H_ex
        args = {'w': rf_freq_mhz * 2 * np.pi}
        
        if b1_rf_mt != 0.0:
            H_1 = (b1_rf_mt * GYRO_E) * (self.sys_rpm.SD[rf_ax] + self.sys_rpm.SA[rf_ax])
            for spin_val, nuc in zip(self.sys_rpm.d_spins, self.sys_rpm.ID):
                gamma_n = GYRO_H1 if spin_val == 0.5 else GYRO_N14
                H_1 += (b1_rf_mt * gamma_n) * nuc[rf_ax]
            for spin_val, nuc in zip(self.sys_rpm.a_spins, self.sys_rpm.IA):
                gamma_n = GYRO_H1 if spin_val == 0.5 else GYRO_N14
                H_1 += (b1_rf_mt * gamma_n) * nuc[rf_ax]
            return [H_0, [H_1, fn]], args
            
        return H_0, args
        
    def _get_initial_state(self, p_val, axis='z'):
        """Internal method to construct the density matrix."""
        axis_map = {'x': [p_val, 0.0, 0.0], 'y': [0.0, p_val, 0.0], 'z': [0.0, 0.0, p_val]}
        p_vec = axis_map.get(axis, [0.0, 0.0, p_val])
        p_d_list = [p_vec for _ in range(len(self.d_spins))]
        p_a_list = [p_vec for _ in range(len(self.a_spins))]
        return self.solver.get_initial_rho(p_d_list, p_a_list, chi_init=self.chi_init)

    def simulate_yield(self, 
                       B0, 
                       t_max, 
                       p_val, pol_axis='z', 
                       theta=0.0, phi=0.0, 
                       b1_rf_mt=0.0, rf_freq_mhz=0.0, rf_ax='z', fn='sin(w * t)'):
        """
        Executes a highly optimized run to extract only final asymptotic yields.
        Returns: (Yield_S, Yield_Tp, Yield_T0, Yield_Tm)
        """
        H, args = self._get_hamiltonian(B0, theta, phi, b1_rf_mt, rf_freq_mhz, rf_ax, fn)
        rho0 = self._get_initial_state(p_val, pol_axis)
        
        e_ops = [self.pop_ops['S_sh'], self.pop_ops['Tp_sh'], self.pop_ops['T0_sh'], self.pop_ops['Tm_sh']]
        times = [0, t_max]
        opts = {'nsteps': 100000}
        
        res = qt.mesolve(H, rho0, times, self.c_ops, e_ops=e_ops, args=args, options=opts)
        
        # Return only the final value [-1] for each shelving state
        return tuple(ex[-1] for ex in res.expect) # type: ignore

    def simulate_dynamics(self, 
                          B0, 
                          times_array, 
                          p_val, pol_axis='z', 
                          theta=0.0, phi=0.0, 
                          b1_rf_mt=0.0, rf_freq_mhz=0.0, rf_ax='z', fn='sin(w * t)'):
        """
        Executes a full time-evolution.
        Returns: (S, Tp, T0, Tm, Yield_S, Yield_Tp, Yield_T0, Yield_Tm) traces.
        """
        H, args = self._get_hamiltonian(B0, theta, phi, b1_rf_mt, rf_freq_mhz, rf_ax, fn)
        rho0 = self._get_initial_state(p_val, pol_axis)
        
        e_ops = [self.pop_ops['S'], self.pop_ops['Tp'], self.pop_ops['T0'], self.pop_ops['Tm'],
                 self.pop_ops['S_sh'], self.pop_ops['Tp_sh'], self.pop_ops['T0_sh'], self.pop_ops['Tm_sh']]
        opts = {'nsteps': 100000}
        
        res = qt.mesolve(H, rho0, times_array, self.c_ops, e_ops=e_ops, args=args, options=opts)
        
        # Return full arrays for all 8 states
        return tuple(np.real(ex) for ex in res.expect)