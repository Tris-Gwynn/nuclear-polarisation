import numpy as np
import qutip as qt
from core import NSpinRPMSystem

class NPolarizedSolver:
    def __init__(self, sys: NSpinRPMSystem):
        self.sys = sys
        dim = self.sys.el_dim
        
        # Electronic States in Product Basis (Top 4x4 block)
        # |0> = |up, up> (Tp)
        # |1> = |up, down>
        # |2> = |down, up>
        # |3> = |down, down> (Tm)
        self.ket_Tp_act = qt.basis(dim, 0)
        self.ket_Tm_act = qt.basis(dim, 3)
        self.ket_S_act  = (qt.basis(dim, 1) - qt.basis(dim, 2)).unit()
        self.ket_T0_act = (qt.basis(dim, 1) + qt.basis(dim, 2)).unit()
        
        # Shelving States (Indices 4 through 7)
        self.ket_S_sh  = qt.basis(dim, 4)
        self.ket_Tp_sh = qt.basis(dim, 5)
        self.ket_T0_sh = qt.basis(dim, 6)
        self.ket_Tm_sh = qt.basis(dim, 7)

        # CISS Shelving State (Index 8) if applicable
        if self.sys.use_ciss:
            self.ket_R_sh = qt.basis(dim, 8)
        
        # Base Electronic Collapse Operators
        self.C_S_el  = self.ket_S_sh @ self.ket_S_act.dag()
        self.C_Tp_el = self.ket_Tp_sh @ self.ket_Tp_act.dag()
        self.C_T0_el = self.ket_T0_sh @ self.ket_T0_act.dag()
        self.C_Tm_el = self.ket_Tm_sh @ self.ket_Tm_act.dag()


    def get_collapse_ops(self, ks, kt, kr=0.0, chi=0.0):
        c_ops = [
            np.sqrt(ks) * self.sys._tensor_op(self.C_S_el, 0),
            np.sqrt(kt) * self.sys._tensor_op(self.C_Tp_el, 0),
            np.sqrt(kt) * self.sys._tensor_op(self.C_T0_el, 0),
            np.sqrt(kt) * self.sys._tensor_op(self.C_Tm_el, 0)
        ]
        
        if self.sys.use_ciss and kr > 0:
            c = np.cos(chi / 2.0)
            s = np.sin(chi / 2.0)
            # Create coherent backscatter state in electronic space
            ket_back_act = c * self.ket_S_act - s * self.ket_T0_act
            C_R_el = self.ket_R_sh @ ket_back_act.dag()
            c_ops.append(np.sqrt(kr) * self.sys._tensor_op(C_R_el, 0))
            
        return c_ops
        
    def get_population_ops(self):
        """Constructs full-space projection operators for active populations."""
        P_S  = self.ket_S_act @ self.ket_S_act.dag()
        P_Tp = self.ket_Tp_act @ self.ket_Tp_act.dag()
        P_T0 = self.ket_T0_act @ self.ket_T0_act.dag()
        P_Tm = self.ket_Tm_act @ self.ket_Tm_act.dag()

        P_S_sh  = self.ket_S_sh @ self.ket_S_sh.dag()
        P_Tp_sh = self.ket_Tp_sh @ self.ket_Tp_sh.dag()
        P_T0_sh = self.ket_T0_sh @ self.ket_T0_sh.dag()
        P_Tm_sh = self.ket_Tm_sh @ self.ket_Tm_sh.dag()

        return {
            'S': self.sys._tensor_op(P_S, 0),
            'Tp': self.sys._tensor_op(P_Tp, 0),
            'T0': self.sys._tensor_op(P_T0, 0),
            'Tm': self.sys._tensor_op(P_Tm, 0),
            'S_sh': self.sys._tensor_op(P_S_sh, 0),
            'Tp_sh': self.sys._tensor_op(P_Tp_sh, 0),
            'T0_sh': self.sys._tensor_op(P_T0_sh, 0),
            'Tm_sh': self.sys._tensor_op(P_Tm_sh, 0)
        }

    def get_initial_rho(self, P_D_list, P_A_list, chi_init=0.0):
        # 1. Pure Singlet Electronic State
        if self.sys.use_ciss:
            c = np.cos(chi_init / 2.0)
            s = np.sin(chi_init / 2.0)
            ket_el_act = c * self.ket_S_act + s * self.ket_T0_act
        else:
            # Pure Singlet
            ket_el_act = self.ket_S_act
            
        rho_el = ket_el_act @ ket_el_act.dag()
        
        # 2. Polarized Nuclear States Helper
        def make_nuc_rho(spins, p_vecs):
            # Return None if the spin list is empty (e.g., no acceptor spins)
            if not spins:
                return None
                
            if len(spins) != len(p_vecs):
                raise ValueError("Number of spins must match number of polarization vectors.")
                
            nuc_rho_list = []
            
            for I, P_vec in zip(spins, p_vecs):
                P_norm = float(np.linalg.norm(P_vec))
                dim = int(2*I + 1)
                rho_mixed = qt.qeye(dim) / dim
                
                if P_norm == 0:
                    nuc_rho_list.append(rho_mixed)
                    continue
                    
                if P_norm > 1.0:
                    P_norm = 1.0  # Constrain to physical bounds
                
                # Extract angles for rotation
                nx, ny, nz = np.array(P_vec) / P_norm
                theta = np.arccos(nz)
                phi = np.arctan2(ny, nx)
                
                # Initialize pure |+I> state and rotate
                pure_state = qt.basis(dim, 0)
                Jy = qt.jmat(I, 'y')
                Jz = qt.jmat(I, 'z')
                
                R_y = (-1j * theta * Jy).expm()
                R_z = (-1j * phi * Jz).expm()
                
                rotated_state = R_z * R_y * pure_state
                rho_pure = rotated_state * rotated_state.dag()
                
                # Mix thermal and pure state, append to list
                nuc_rho_list.append((1 - P_norm) * rho_mixed + P_norm * rho_pure)
                
            # Tensor product fuses all the nuclei in this specific list together
            return qt.tensor(*nuc_rho_list)

        # 3. Assemble the Full System Density Matrix
        final_rho_list = [rho_el]
        
        # Generate and append donor nuclear states (if any exist)
        rho_D = make_nuc_rho(self.sys.d_spins, P_D_list)
        if rho_D is not None:
            final_rho_list.append(rho_D)
            
        # Generate and append acceptor nuclear states (if any exist)
        rho_A = make_nuc_rho(self.sys.a_spins, P_A_list)
        if rho_A is not None:
            final_rho_list.append(rho_A)
            
        return qt.tensor(*final_rho_list)