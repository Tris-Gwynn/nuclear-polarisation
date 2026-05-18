import numpy as np
import qutip as qt
import scipy.sparse as sp
from constants import GYRO_E

class NSpinRPMSystem:
    def __init__(self, d_spins=[0.5], a_spins=[]):
        self.d_spins = d_spins
        self.a_spins = a_spins
        
        # 8D Electronic Space: 4 Active (Product Basis) + 4 Shelving (S, Tp, T0, Tm)
        self.el_dim = 8 
        self.nuc_dims = [int(2*s + 1) for s in d_spins + a_spins]
        self.dims = [self.el_dim] + self.nuc_dims
        
        # --- Spin Operators (Active 4x4 padded to 8x8) ---
        S_x, S_y, S_z = 0.5 * qt.sigmax(), 0.5 * qt.sigmay(), 0.5 * qt.sigmaz()
        I2 = qt.qeye(2)
        
        # Donor & Acceptor Electrons in 4x4 product basis
        SD_x_4, SD_y_4, SD_z_4 = qt.tensor(S_x, I2), qt.tensor(S_y, I2), qt.tensor(S_z, I2)
        SA_x_4, SA_y_4, SA_z_4 = qt.tensor(I2, S_x), qt.tensor(I2, S_y), qt.tensor(I2, S_z)
        
        def pad8(op4):
            return qt.Qobj(sp.block_diag([op4.full(), np.zeros((4,4))], format="csr"))
            
        self.SD = {'x': self._tensor_op(pad8(SD_x_4), 0), 
                   'y': self._tensor_op(pad8(SD_y_4), 0), 
                   'z': self._tensor_op(pad8(SD_z_4), 0)}
                   
        self.SA = {'x': self._tensor_op(pad8(SA_x_4), 0), 
                   'y': self._tensor_op(pad8(SA_y_4), 0), 
                   'z': self._tensor_op(pad8(SA_z_4), 0)}
        
        # --- Nuclear Operators ---
        self.ID = []
        self.IA = []
        
        idx = 1
        for s in self.d_spins:
            self.ID.append({
                'x': self._tensor_op(qt.jmat(s, 'x'), idx),
                'y': self._tensor_op(qt.jmat(s, 'y'), idx),
                'z': self._tensor_op(qt.jmat(s, 'z'), idx)
            })
            idx += 1
            
        for s in self.a_spins:
            self.IA.append({
                'x': self._tensor_op(qt.jmat(s, 'x'), idx),
                'y': self._tensor_op(qt.jmat(s, 'y'), idx),
                'z': self._tensor_op(qt.jmat(s, 'z'), idx)
            })
            idx += 1
            
        # Shelving Projectors (Full Space)
        self.P_shelf_S  = self._tensor_op(qt.fock_dm(8, 4), 0)
        self.P_shelf_Tp = self._tensor_op(qt.fock_dm(8, 5), 0)
        self.P_shelf_T0 = self._tensor_op(qt.fock_dm(8, 6), 0)
        self.P_shelf_Tm = self._tensor_op(qt.fock_dm(8, 7), 0)

    def _tensor_op(self, op, pos):
        """Places a local operator into the full Hilbert space."""
        op_list = [qt.qeye(d) for d in self.dims]
        op_list[pos] = op
        return qt.tensor(op_list)

# --- N-Spin Hamiltonians ---
def get_rotation_matrix(theta, phi):
    r_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    r_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return r_z @ r_y

def get_n_spin_anisotropic_hyperfine(sys: NSpinRPMSystem, A_tensor_D_list, A_tensor_A_list, theta=0.0, phi=0.0):
    """
    Constructs the anisotropic hyperfine Hamiltonian.
    Incorporates the hyperfine angle directly into the tensor rotation.
    """
    R = get_rotation_matrix(theta, phi)
    H_accum = 0
    dirs = ['x', 'y', 'z']
    
    # Process Donor Tensors
    for k, A_matrix in enumerate(A_tensor_D_list):
        A_rot = R @ A_matrix @ R.T
        for i, d1 in enumerate(dirs):
            for j, d2 in enumerate(dirs):
                if A_rot[i, j] != 0:
                    H_accum += (A_rot[i, j] * GYRO_E) * sys.SD[d1] * sys.ID[k][d2]
                    
    # Process Acceptor Tensors
    for k, A_matrix in enumerate(A_tensor_A_list):
        A_rot = R @ A_matrix @ R.T
        for i, d1 in enumerate(dirs):
            for j, d2 in enumerate(dirs):
                if A_rot[i, j] != 0:
                    H_accum += (A_rot[i, j] * GYRO_E) * sys.SA[d1] * sys.IA[k][d2]
                    
    if isinstance(H_accum, int) and H_accum == 0:
        return qt.tensor([qt.qzero(d) for d in sys.dims])
    return H_accum

def get_n_spin_zeeman(sys: NSpinRPMSystem, B0, theta=0.0, phi=0.0):
    """
    Zeeman Hamiltonian with orientational dependence.
    """
    Bx = B0 * np.sin(theta) * np.cos(phi)
    By = B0 * np.sin(theta) * np.sin(phi)
    Bz = B0 * np.cos(theta)
    
    omega_x = Bx * GYRO_E
    omega_y = By * GYRO_E
    omega_z = Bz * GYRO_E
    
    return (omega_x * (sys.SD['x'] + sys.SA['x']) + 
            omega_y * (sys.SD['y'] + sys.SA['y']) + 
            omega_z * (sys.SD['z'] + sys.SA['z']))

def get_n_spin_dipolar(sys: NSpinRPMSystem, D_tensor, theta=0.0, phi=0.0):
    """
    Electron-Electron Dipolar Coupling from a full 3x3 tensor.
    theta and phi represent the global rotation of the molecule.
    """
    R = get_rotation_matrix(theta, phi)
    D_rot = R @ D_tensor @ R.T
    
    H_accum = 0
    dirs = ['x', 'y', 'z']
    for i, d1 in enumerate(dirs):
        for j, d2 in enumerate(dirs):
            if D_rot[i, j] != 0:
                H_accum += (D_rot[i, j] * GYRO_E) * sys.SD[d1] * sys.SA[d2]
                
    if isinstance(H_accum, int) and H_accum == 0:
        return qt.tensor([qt.qzero(d) for d in sys.dims])
    return H_accum

def get_n_spin_exchange(sys: NSpinRPMSystem, J):
    """
    Isotropic Electron-Electron Exchange Coupling.
    """
    dot = (sys.SD['x'] * sys.SA['x'] + 
           sys.SD['y'] * sys.SA['y'] + 
           sys.SD['z'] * sys.SA['z'])
           
    ident = qt.tensor([qt.qeye(d) for d in sys.dims])
    
    return -J * (2 * dot + 0.5 * ident)