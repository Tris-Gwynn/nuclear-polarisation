import numpy as np
import qutip as qt
from core import NSpinRPMSystem

class NPolarizedSolver:
    def __init__(self, sys: NSpinRPMSystem):
        self.sys = sys
        
        # Electronic States in Product Basis (Top 4x4 block)
        # |0> = |up, up> (Tp)
        # |1> = |up, down>
        # |2> = |down, up>
        # |3> = |down, down> (Tm)
        ket_Tp_act = qt.basis(8, 0)
        ket_Tm_act = qt.basis(8, 3)
        ket_S_act  = (qt.basis(8, 1) - qt.basis(8, 2)).unit()
        ket_T0_act = (qt.basis(8, 1) + qt.basis(8, 2)).unit()
        
        # Shelving States (Indices 4 through 7)
        ket_S_sh  = qt.basis(8, 4)
        ket_Tp_sh = qt.basis(8, 5)
        ket_T0_sh = qt.basis(8, 6)
        ket_Tm_sh = qt.basis(8, 7)
        
        # Base Electronic Collapse Operators
        self.C_S_el  = ket_S_sh * ket_S_act.dag()
        self.C_Tp_el = ket_Tp_sh * ket_Tp_act.dag()
        self.C_T0_el = ket_T0_sh * ket_T0_act.dag()
        self.C_Tm_el = ket_Tm_sh * ket_Tm_act.dag()

    def get_collapse_ops(self, ks, kt):
        return [
            np.sqrt(ks) * self.sys._tensor_op(self.C_S_el, 0),
            np.sqrt(kt) * self.sys._tensor_op(self.C_Tp_el, 0),
            np.sqrt(kt) * self.sys._tensor_op(self.C_T0_el, 0),
            np.sqrt(kt) * self.sys._tensor_op(self.C_Tm_el, 0)
        ]
        
    def get_initial_rho(self, P_D_list, P_A_list):
        # 1. Pure Singlet Electronic State
        ket_S_act = (qt.basis(8, 1) - qt.basis(8, 2)).unit()
        rho_el = ket_S_act * ket_S_act.dag()
        
        rho_list = [rho_el]
        
        # 2. Polarized Nuclear States
        def make_nuc_rho(I, P):
            dim = int(2*I + 1)
            rho_mixed = qt.qeye(dim) / dim
            if P == 0:
                return rho_mixed
            
            # Index 0 is spin up (+I), Index (dim-1) is spin down (-I)
            pure_idx = 0 if P > 0 else dim - 1
            rho_pure = qt.fock_dm(dim, pure_idx)
            
            return (1 - abs(P)) * rho_mixed + abs(P) * rho_pure

        for I, P in zip(self.sys.d_spins, P_D_list):
            rho_list.append(make_nuc_rho(I, P))
            
        for I, P in zip(self.sys.a_spins, P_A_list):
            rho_list.append(make_nuc_rho(I, P))
            
        print(qt.tensor(rho_list))
        return qt.tensor(rho_list)
    
