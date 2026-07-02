"""
Scenario 4: Nuclear Topology (Donor vs Acceptor Distribution)
Evaluates specific distributions of equivalent isotropic nuclei across the donor and acceptor.
Outputs a 2x2 subplot grid comparing the configurations.
"""
import time
import pickle
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from RPM_Experiment_Builder import RPMBuilder

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
a_iso = (0.2 + 0.03 + 1.0) / 3.0
A_iso_base = np.array([
    [ a_iso, 0.0, 0.0],
    [ 0.0, a_iso, 0.0],
    [ 0.0, 0.0, a_iso]
])

# Define the nuclear topologies: (n_donor, n_acceptor)
TOPOLOGIES = {
    '1D_0A': (1, 0),
    '0D_1A': (0, 1),
    '1D_1A': (1, 1),
    '2D_2A': (2, 2)
}

J_EX = 0.0
D_TENSOR = None

T_MAX = 5.0
P_LOW = 0.0
P_HIGH = 1.0
K_S, K_T = 1.0, 1.0

B0_MIN, B0_MAX, B0_POINTS = 1e-3, 10.0, 100
B0_ARRAY = np.logspace(np.log10(B0_MIN), np.log10(B0_MAX), B0_POINTS)

def get_discrete_angles():
    return [
        (0.0, 0.0),             # Z-axis
        (np.pi/2, 0.0),         # X-axis
        (np.pi/2, np.pi/2),     # Y-axis
        (np.pi/4, np.pi/4)      # 45-degree off-axis
    ]

def compute_mary_curve(args):
    topo_name, n_donor, n_acceptor, theta, phi = args
    
    # Construct the specific D/A arrays
    d_spins = [0.5] * n_donor
    a_spins = [0.5] * n_acceptor
    a_tensor_d = [A_iso_base] * n_donor
    a_tensor_a = [A_iso_base] * n_acceptor
    
    builder = RPMBuilder(d_spins=d_spins, a_spins=a_spins,
                         a_tensor_d=a_tensor_d, a_tensor_a=a_tensor_a,
                         d_tensor=D_TENSOR, j_ex=J_EX,
                         k_s=K_S, k_t=K_T)

    y_low, y_x, y_y, y_z = [], [], [], []
    
    for B0 in B0_ARRAY:
        y_low.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_LOW, pol_axis='z', theta=theta, phi=phi)[0])
        y_x.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='x', theta=theta, phi=phi)[0])
        y_y.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='y', theta=theta, phi=phi)[0])
        y_z.append(builder.yield_(B0=B0, t_max=T_MAX, p_val=P_HIGH, pol_axis='z', theta=theta, phi=phi)[0])

    return topo_name, theta, phi, y_low, y_x, y_y, y_z

if __name__ == '__main__':
    start_time = time.perf_counter()
    angles = get_discrete_angles()
    
    # Build argument list for all topologies and angles
    all_args = []
    for topo_name, (n_d, n_a) in TOPOLOGIES.items():
        for t, p in angles:
            all_args.append((topo_name, n_d, n_a, t, p))

    print(f"Executing {len(all_args)} sweeps evaluating Donor/Acceptor topologies...")

    cores = min(16, mp.cpu_count())
    with mp.Pool(processes=cores) as pool:
        results = pool.map(compute_mary_curve, all_args)

    # Organize data
    data_dict = {
        'metadata': {'B0_ARRAY': B0_ARRAY, 'TOPOLOGIES': TOPOLOGIES, 'a_iso': a_iso},
        'results': {topo_name: [] for topo_name in TOPOLOGIES.keys()}
    }

    for res in results:
        topo_name, theta, phi, y_low, y_x, y_y, y_z = res
        data_dict['results'][topo_name].append({
            'theta': theta, 'phi': phi,
            'y_low': y_low, 'y_x': y_x, 'y_y': y_y, 'y_z': y_z
        })

    fname = 'scenario4_data.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(data_dict, f)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    lw = 1.0
    alpha_val = 0.8

    for ax, (topo_name, counts) in zip(axes, TOPOLOGIES.items()):
        orientations = data_dict['results'][topo_name]
        
        for i, item in enumerate(orientations):
            y_low, y_x, y_y, y_z = item['y_low'], item['y_x'], item['y_y'], item['y_z']
            
            lbl_low = rf'Unpolarized' if i == 0 else None
            lbl_x   = rf'Pol X' if i == 0 else None
            lbl_y   = rf'Pol Y' if i == 0 else None
            lbl_z   = rf'Pol Z' if i == 0 else None

            ax.plot(B0_ARRAY, y_low, color='black', lw=lw, ls='--', alpha=alpha_val, label=lbl_low)
            ax.plot(B0_ARRAY, y_x,   color='C0', lw=lw, alpha=alpha_val, label=lbl_x)
            ax.plot(B0_ARRAY, y_y,   color='C1', lw=lw, alpha=alpha_val, label=lbl_y)
            ax.plot(B0_ARRAY, y_z,   color='C2', lw=lw, alpha=alpha_val, label=lbl_z)

        ax.set_xscale('log')
        ax.set_title(f'Topology: {topo_name} ({counts[0]} Donor, {counts[1]} Acceptor)')
        
        if ax in [axes[2], axes[3]]:
            ax.set_xlabel(r'Static Magnetic Field $B_0$ (mT)')
        if ax in [axes[0], axes[2]]:
            ax.set_ylabel(r'Asymptotic Singlet Yield ($\Phi_S$)')
        if ax == axes[0]:
            ax.legend(loc='lower right', frameon=False, fontsize='small')

    plt.suptitle(f'MARY Curve Degradation vs Nuclear Topology (Isotropic $a_{{iso}} = {a_iso:.3f}$ mT)', fontsize=14)
    plt.tight_layout()
    
    out_fname = 'scenario4_topology.png'
    plt.savefig(out_fname, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Scenario 4 complete in {time.perf_counter()-start_time:.2f}s. Saved to {out_fname}")