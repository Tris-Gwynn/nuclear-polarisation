import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST be before importing pyplot
import matplotlib.pyplot as plt
import qutip as qt
import time
import multiprocessing
# ... the rest of your imports stay the same ...
from concurrent.futures import ProcessPoolExecutor, as_completed
from core import (NSpinRPMSystem, 
                  get_n_spin_anisotropic_hyperfine, 
                  get_n_spin_zeeman,
                  get_n_spin_dipolar,
                  get_n_spin_exchange)
from solver import NPolarizedSolver

# --- 1. CONFIGURATION ---
D_SPINS = [0.5] # One spin on the donor
A_SPINS = []    # Zero spins on the acceptor

TrpH_N1 = np.array([[-0.0529384, 0.0586559, -0.0460243], [0.0586559, 0.564457, -0.564786], [-0.0460243, -0.564786, 0.453143]])
TrpH_H1 = np.array([[-1.00098, 0.206185, 0.193279], [0.206185, -0.441571, 0.307287], [0.193279, 0.307287, -0.352472]])
FAD_N5 = np.array([[-0.0994543, 0.002869, 0.0], [0.002869, -0.087455, 0.0], [0.0, 0.0, 1.75690]])
FAD_N10 = np.array([[-0.0148798, -0.00205738, 0.0], [-0.00205738, -0.0236651, 0.0], [0.0, 0.0, 0.604580]])

# FIX: Lengths must match len(D_SPINS) and len(A_SPINS)
A_TENSOR_D_LIST = [TrpH_H1] 
A_TENSOR_A_LIST = []

B0 = 1
# Example Angles for Zeeman
THETA = np.pi / 2 
PHI = 0.0

# Example parameters for Dipolar/Exchange (set to 0 to disable)
D_TENSOR = np.array([
    [ 0.030687, -0.338269,  0.136643],
    [-0.338269, -0.218231,  0.195874],
    [ 0.136643,  0.195874,  0.187544]
])
J_EX = 0.0

k_S = 10.0
k_T = 10.0
T_MAX = 5.0 / k_S 

times = np.linspace(0, T_MAX, 200) 
polarizations = np.linspace(-1.0, 1.0, 41) 

# --- 2. STATIC SYSTEM SETUP ---
print("Building static Hamiltonians...")
sys = NSpinRPMSystem(d_spins=D_SPINS, a_spins=A_SPINS)
solver = NPolarizedSolver(sys)

H_hf = get_n_spin_anisotropic_hyperfine(sys, A_TENSOR_D_LIST, A_TENSOR_A_LIST, theta=0.0, phi=0.0)
H_z = get_n_spin_zeeman(sys, B0, theta=THETA, phi=PHI)
H_dip = get_n_spin_dipolar(sys, D_TENSOR, theta=THETA, phi=PHI)
H_ex = get_n_spin_exchange(sys, J_EX)

H_tot = H_hf + H_z + H_dip + H_ex
c_ops = solver.get_collapse_ops(k_S, k_T)
e_ops = [sys.P_shelf_S, sys.P_shelf_Tp, sys.P_shelf_T0, sys.P_shelf_Tm]

# --- 3. PARALLEL EXECUTION FUNCTION ---
def compute_yields(p):
    """Worker function to be executed across multiple CPU cores."""
    
    # FIX: Dynamically size the polarization lists based on the configuration.
    # This automatically assigns 'p' to all donor nuclei and '0.0' to all acceptor nuclei.
    P_D_LIST = [p] * len(D_SPINS)
    P_A_LIST = [0.0] * len(A_SPINS)
    
    rho0 = solver.get_initial_rho(P_D_LIST, P_A_LIST)
    result = qt.mesolve(H_tot, rho0, times, c_ops, e_ops=e_ops)
    
    # Return the terminal yields for S, Tp, T0, Tm
    return (
        np.real(result.expect[0][-1]),
        np.real(result.expect[1][-1]),
        np.real(result.expect[2][-1]),
        np.real(result.expect[3][-1])
    )


# --- 4. EXECUTE SWEEP ---
if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    total_points = len(polarizations)
    print(f"Sweeping {total_points} points across {cores} CPU cores...")
    
    start_time = time.time()
    
    # Pre-allocate a list to guarantee the results stay in the exact same order as the polarizations
    results = [None] * total_points 
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        # Submit all jobs to the cores and keep track of their original index
        future_to_index = {
            executor.submit(compute_yields, p): i 
            for i, p in enumerate(polarizations)
        }
        
        completed_count = 0
        # As each core finishes its calculation, this loop triggers
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result() # Slot the result back into its correct position
            
            completed_count += 1
            # Print progress dynamically, forcing the terminal to flush the output immediately
            print(f"Progress: {completed_count}/{total_points} completed...", flush=True)
            
    print(f"Sweep completed in {time.time() - start_time:.2f} seconds.")

    # Unpack the results
    final_yields_S  = [r[0] for r in results]
    final_yields_Tp = [r[1] for r in results]
    final_yields_T0 = [r[2] for r in results]
    final_yields_Tm = [r[3] for r in results]
    
 # --- 5. SAVING DATA & PLOTTING ---
    
    # Save the raw data arrays to a compressed numpy file
    data_filename = 'polarization_sweep_data.npz'
    np.savez(data_filename, 
             polarizations=polarizations, 
             yields_S=final_yields_S, 
             yields_Tp=final_yields_Tp, 
             yields_T0=final_yields_T0, 
             yields_Tm=final_yields_Tm)
    print(f"Raw data saved to {data_filename}")

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Terminal Yields vs. Nuclear Polarization\n(Both TrpH Nuclei Polarized uniformly)")

    plt.plot(polarizations, final_yields_S,  'k--', linewidth=2, label='Singlet Yield')
    plt.plot(polarizations, final_yields_Tp, 'r-',  linewidth=1.5, label='Tp Yield')
    plt.plot(polarizations, final_yields_T0, 'g-',  linewidth=1.5, label='T0 Yield')
    plt.plot(polarizations, final_yields_Tm, 'b-',  linewidth=1.5, label='Tm Yield')

    plt.xlabel(r"Degree of Polarization ($P_z$)")
    plt.ylabel(r"Fractional Yield ($t \to \infty$)")
    plt.xlim(-1.0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5) 
    plt.tight_layout()
    
    # Save the figure instead of showing it
    plot_filename = 'polarization_sweep_plot.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close() # Free up memory
    print(f"Plot saved to {plot_filename}")