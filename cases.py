import numpy as np

# =============================================================================
# HYPERFINE TENSOR LIBRARY (Values in mT)
# =============================================================================
# Principal components extracted from Hiscock et al. 2016 

TENSORS = {
    'FAD': {
        'N5': np.array([
            [-0.0989,  0.0039,  0.0   ],
            [ 0.0039, -0.0881,  0.0   ],
            [ 0.0   ,  0.0   ,  1.7569]
        ]),
        'N10': np.array([
            [-0.0190, -0.0048,  0.0   ],
            [-0.0048, -0.0196,  0.0   ],
            [ 0.0   ,  0.0   ,  0.6046]
        ]),
        'H6': np.array([
            [-0.2569, -0.1273,  0.0   ],
            [-0.1273, -0.4711,  0.0   ],
            [ 0.0   ,  0.0   , -0.4336]
        ]),
        # Isotropic averages for methyl groups (Off-diagonals are strictly 0)
        'H8_iso': np.array([
            [ 0.4399,  0.0   ,  0.0   ],
            [ 0.0   ,  0.4399,  0.0   ],
            [ 0.0   ,  0.0   ,  0.4399]
        ]),
        'Hbeta_iso': np.array([
            [ 0.4070,  0.0   ,  0.0   ],
            [ 0.0   ,  0.4070,  0.0   ],
            [ 0.0   ,  0.0   ,  0.4070]
        ])
    },
    'TrpH': {
        'N1': np.array([
            [-0.0336,  0.0924, -0.1354],
            [ 0.0924,  0.3303, -0.5318],
            [-0.1354, -0.5318,  0.6680]
        ]),
        'H1': np.array([
            [-0.9920, -0.2091, -0.2003],
            [-0.2091, -0.2631,  0.2803],
            [-0.2003,  0.2803, -0.5398]
        ]),
        'H2': np.array([
            [-0.2843,  0.1757,  0.1525],
            [ 0.1757, -0.2798,  0.0975],
            [ 0.1525,  0.0975, -0.2699]
        ]),
        'H4': np.array([
            [-0.5596, -0.1956, -0.1657],
            [-0.1956, -0.4020,  0.0762],
            [-0.1657,  0.0762, -0.5021]
        ]),
        'H6': np.array([
            [ 0.0622, -0.3100, -0.0297],
            [-0.3100, -0.2083, -0.0494],
            [-0.0297, -0.0494,  0.2642]
        ]),
        'H7': np.array([
            [-0.1541, -0.2777,  0.0864],
            [-0.2777, -0.3636, -0.0594],
            [ 0.0864, -0.0594, -0.3770]
        ]),
        'Hbeta1': np.array([
            [-0.0453,  1.5575,  0.0988],
            [ 1.5575,  1.6046, -0.0456],
            [ 0.0988, -0.0456,  1.6752]
        ])
    }
}

# Rotation matrix mapping TrpH frame onto FAD frame
R_TRP_TO_FAD = np.array([
    [-0.3454,  0.6869, -0.6395],
    [-0.8269,  0.0995,  0.5535],
    [ 0.4438,  0.7199,  0.5336]
])


def get_nuclear_case(case_name, **kwargs):
    """
    Returns the spin vectors and coupled hyperfine tensors for a given RPM model.
    """
    # Baseline coupling for toy models (mT)
    A_BASE = 0.05684
    
    if case_name == 'toy_1_nuc':
        loc = kwargs.get('location', 'donor')
        anisotropy = kwargs.get('anisotropy', 'anisotropic')
        
        if anisotropy == 'isotropic':
            base_tensor = np.diag([A_BASE, A_BASE, A_BASE])
        else:
            base_tensor = np.diag([A_BASE/10.0, A_BASE/10.0, A_BASE])
            
        if loc == 'donor':
            return {'D_SPINS': [0.5], 'A_SPINS': [], 'A_TENSOR_D_LIST': [base_tensor], 'A_TENSOR_A_LIST': []}
        else:
            return {'D_SPINS': [], 'A_SPINS': [0.5], 'A_TENSOR_D_LIST': [], 'A_TENSOR_A_LIST': [base_tensor]}

    elif case_name == 'toy_2_nuc':
        anisotropy = kwargs.get('anisotropy', 'anisotropic')
        
        if anisotropy == 'isotropic':
            base_tensor = np.diag([A_BASE, A_BASE, A_BASE])
        else:
            base_tensor = np.diag([A_BASE/10.0, A_BASE/10.0, A_BASE])
            
        # Opposed tensors to break spatial symmetry
        return {
            'D_SPINS': [0.5], 
            'A_SPINS': [0.5], 
            'A_TENSOR_D_LIST': [base_tensor], 
            'A_TENSOR_A_LIST': [-base_tensor]
        }

    elif case_name == '4_real_nuc':
        # Select highest-coupling nuclei: FAD N5, N10 and TrpH N1, H1
        D_SPINS = [1.0, 1.0] # 14N, 14N
        A_SPINS = [1.0, 0.5] # 14N, 1H
        
        A_TENSOR_D_LIST = [
            TENSORS['FAD']['N5'], 
            TENSORS['FAD']['N10']
        ]
        
        # Acceptor tensors must be rotated into the global (FAD) frame
        raw_acceptor = [
            TENSORS['TrpH']['N1'], 
            TENSORS['TrpH']['H1']
        ]
        A_TENSOR_A_LIST = [R_TRP_TO_FAD @ t @ R_TRP_TO_FAD.T for t in raw_acceptor]
        
        return {
            'D_SPINS': D_SPINS, 
            'A_SPINS': A_SPINS, 
            'A_TENSOR_D_LIST': A_TENSOR_D_LIST, 
            'A_TENSOR_A_LIST': A_TENSOR_A_LIST
        }

    elif case_name == '8_real_nuc':
        # 8-Nuclei Intermediate Model (4 Donor, 4 Acceptor)
        # FAD: N5, N10, H6, H8_iso
        # TrpH: N1, H1, H2, H4
        D_SPINS = [1.0, 1.0, 0.5, 0.5] 
        A_SPINS = [1.0, 0.5, 0.5, 0.5] 
        
        A_TENSOR_D_LIST = [
            TENSORS['FAD']['N5'], 
            TENSORS['FAD']['N10'], 
            TENSORS['FAD']['H6'],
            TENSORS['FAD']['H8_iso']
        ]
        
        # Acceptor tensors must be rotated into the global (FAD) frame
        raw_acceptor = [
            TENSORS['TrpH']['N1'], 
            TENSORS['TrpH']['H1'],
            TENSORS['TrpH']['H2'],
            TENSORS['TrpH']['H4']
        ]
        A_TENSOR_A_LIST = [R_TRP_TO_FAD @ t @ R_TRP_TO_FAD.T for t in raw_acceptor]
        
        return {
            'D_SPINS': D_SPINS, 
            'A_SPINS': A_SPINS, 
            'A_TENSOR_D_LIST': A_TENSOR_D_LIST, 
            'A_TENSOR_A_LIST': A_TENSOR_A_LIST
        }

    elif case_name == 'full_real_nuc':
        # 14-Nuclei Hiscock Model (7 Donor, 7 Acceptor)
        D_SPINS = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        A_SPINS = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        A_TENSOR_D_LIST = [
            TENSORS['FAD']['N5'], TENSORS['FAD']['N10'], TENSORS['FAD']['H6'],
            TENSORS['FAD']['H8_iso'], TENSORS['FAD']['H8_iso'], TENSORS['FAD']['H8_iso'],
            TENSORS['FAD']['Hbeta_iso']
        ]
        
        raw_acceptor = [
            TENSORS['TrpH']['N1'], TENSORS['TrpH']['H1'], TENSORS['TrpH']['H2'],
            TENSORS['TrpH']['H4'], TENSORS['TrpH']['H6'], TENSORS['TrpH']['H7'],
            TENSORS['TrpH']['Hbeta1']
        ]
        A_TENSOR_A_LIST = [R_TRP_TO_FAD @ t @ R_TRP_TO_FAD.T for t in raw_acceptor]
        
        return {
            'D_SPINS': D_SPINS, 
            'A_SPINS': A_SPINS, 
            'A_TENSOR_D_LIST': A_TENSOR_D_LIST, 
            'A_TENSOR_A_LIST': A_TENSOR_A_LIST
        }
        
    else:
        raise ValueError(f"Case '{case_name}' is not defined.")