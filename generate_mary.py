"""
Generates and saves the MARY curve data over a spherical grid.
"""
import time
import pickle
import multiprocessing as mp
import numpy as np

from RPM_Experiment_Builder import RPMBuilder
from plot_mary import plot_data  # Import the plotting function

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
a0 = 0.5  # mT
Axx, Ayy, Azz = 0.2, 0.03, 1.0

A = np.array([
            [ Axx, 0.0, 0.0],
            [ 0.0, Ayy, 0.0],
            [ 0.0, 0.0, Azz]
            ])

D_SPINS = [0.5]
A_SPINS = []
A_TENSOR_D = [A]
A_TENSOR_A = []

J_EX = 0.0
D_TENSOR = None

T_MAX = 5.0
P_LOW = 0.0
P_HIGH = 1.0

K_S, K_T = 1.0, 1.0

B0_MIN, B0_MAX, B0_POINTS = 1e-3, 10.0, 100
B0_ARRAY = np.logspace(np.log10(B0_MIN), np.log10(B0_MAX), B0_POINTS)

def generate_angle_grid():
    """
    Returns 4 discrete orientations: Z-axis, X-axis, Y-axis, and (pi/4, pi/4).
    """
    return [
        (0.0, 0.0),             # Z-axis
        (np.pi/2, 0.0),         # X-axis
        (np.pi/2, np.pi/2),     # Y-axis
        (np.pi/4, np.pi/4)      # Theta=45 deg, Phi=45 deg
    ]

def compute_mary_curve(angle):
    theta, phi = angle
    builder = RPMBuilder(d_spins=D_SPINS, a_spins=A_SPINS,
                         a_tensor_d=A_TENSOR_D, a_tensor_a=A_TENSOR_A,
                         d_tensor=D_TENSOR, j_ex=J_EX,
                         k_s=K_S, k_t=K_T)

    y_low, y_x, y_y, y_z = [], [], [], []
    
    for B0 in B0_ARRAY:
        yld_low = builder.yield_(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis='z', theta=theta, phi=phi)[0]
        yld_x   = builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=theta, phi=phi)[0]
        yld_y   = builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=theta, phi=phi)[0]
        yld_z   = builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=theta, phi=phi)[0]
        
        y_low.append(yld_low)
        y_x.append(yld_x)
        y_y.append(yld_y)
        y_z.append(yld_z)

    return y_low, y_x, y_y, y_z

if __name__ == '__main__':
    start_time = time.perf_counter()
    
    angles = generate_angle_grid()
    total_orientations = len(angles)
    print(f"Sweeping {B0_POINTS} magnetic field points over {total_orientations} orientations...")

    cores = 10
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_curve, angles)

    # Pack data into a dictionary
    data_dict = {
        'metadata': {
            'B0_ARRAY': B0_ARRAY,
            'P_LOW': P_LOW,
            'P_HIGH': P_HIGH
        },
        'orientations': []
    }

    for (theta, phi), (y_low, y_x, y_y, y_z) in zip(angles, results):
        data_dict['orientations'].append({
            'theta': theta,
            'phi': phi,
            'y_low': np.array(y_low),
            'y_x': np.array(y_x),
            'y_y': np.array(y_y),
            'y_z': np.array(y_z)
        })

    fname = 'mary_data.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"Data generation complete in {time.perf_counter()-start_time:.2f}s. Saved dictionary to {fname}")

    # Trigger plotting automatically
    print("Initiating plot rendering...")
    plot_data(fname)