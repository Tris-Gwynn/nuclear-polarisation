# %%
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import ipywidgets as widgets
from IPython.display import display, clear_output

from core import (
    NSpinRPMSystem, 
    get_n_spin_anisotropic_hyperfine, 
    get_n_spin_zeeman, 
    get_n_spin_dipolar, 
    get_n_spin_exchange
)
from solver import NPolarizedSolver
from constants import GYRO_E

# =============================================================================
# 1. DYNAMIC UI DEFINITION
# =============================================================================
style = {'description_width': 'initial'}

# Storage for dynamic widget references
donor_widgets_list = []
acceptor_widgets_list = []

def create_nucleus_block(title):
    """Generates a block of widgets for a single nucleus."""
    w_spin = widgets.FloatText(value=0.5, step=0.5, description='Spin (I):', style=style)
    w_ax = widgets.FloatText(value=1.0, step=0.1, description='A_X (MHz):', style=style)
    w_ay = widgets.FloatText(value=1.0, step=0.1, description='A_Y (MHz):', style=style)
    w_az = widgets.FloatText(value=0.5, step=0.1, description='A_Z (MHz):', style=style)
    w_pol = widgets.Text(value="0.0, 0.0, 0.0", description='Pol [x,y,z]:', style=style)
    
    ui_box = widgets.VBox([
        widgets.HTML(f"<b>{title}</b>"),
        w_spin, w_ax, w_ay, w_az, w_pol,
        widgets.HTML("<hr>")
    ])
    
    # Return the dictionary of widget objects for later extraction, and the UI layout
    data_dict = {'spin': w_spin, 'ax': w_ax, 'ay': w_ay, 'az': w_az, 'pol': w_pol}
    return data_dict, ui_box

# --- Dynamic Tab 1: Donor Nuclei ---
container_donor = widgets.VBox([])
btn_add_donor = widgets.Button(description="+ Add Donor Nucleus", button_style='info')

def add_donor_event(b):
    count = len(donor_widgets_list) + 1
    data_dict, ui_box = create_nucleus_block(f"Donor Nucleus {count}")
    donor_widgets_list.append(data_dict)
    container_donor.children = tuple(list(container_donor.children) + [ui_box])

btn_add_donor.on_click(add_donor_event)
tab_donor = widgets.VBox([btn_add_donor, container_donor])

# --- Dynamic Tab 2: Acceptor Nuclei ---
container_acceptor = widgets.VBox([])
btn_add_acceptor = widgets.Button(description="+ Add Acceptor Nucleus", button_style='info')

def add_acceptor_event(b):
    count = len(acceptor_widgets_list) + 1
    data_dict, ui_box = create_nucleus_block(f"Acceptor Nucleus {count}")
    acceptor_widgets_list.append(data_dict)
    container_acceptor.children = tuple(list(container_acceptor.children) + [ui_box])

btn_add_acceptor.on_click(add_acceptor_event)
tab_acceptor = widgets.VBox([btn_add_acceptor, container_acceptor])

# --- Pre-populate with one Donor for convenience ---
add_donor_event(None)

# --- Tab 3: Electron-Electron Couplings ---
w_jex = widgets.FloatText(value=0.0, step=0.1, description='Exchange J (MHz):', style=style)
w_dx = widgets.FloatText(value=0.0, step=0.1, description='Dipolar X (MHz):', style=style)
w_dy = widgets.FloatText(value=0.0, step=0.1, description='Dipolar Y (MHz):', style=style)
w_dz = widgets.FloatText(value=0.0, step=0.1, description='Dipolar Z (MHz):', style=style)
tab_ee = widgets.VBox([w_jex, widgets.HTML("<b>Dipolar Tensor (Diagonal)</b>"), w_dx, w_dy, w_dz])

# --- Tab 4: Magnetic Fields ---
w_b0 = widgets.FloatSlider(value=0.05, min=0.0, max=5.0, step=0.01, description='B0 (mT):', style=style)
w_theta = widgets.FloatSlider(value=0.0, min=0.0, max=np.pi, step=0.01, description='Theta (rad):', style=style)
w_phi = widgets.FloatSlider(value=np.pi/4, min=0.0, max=2*np.pi, step=0.01, description='Phi (rad):', style=style)
w_b1 = widgets.FloatSlider(value=0.0, min=0.0, max=2.0, step=0.05, description='RF B1 (mT):', style=style)
w_rf_freq = widgets.FloatText(value=0.017, description='RF Freq (MHz):', style=style)
tab_fields = widgets.VBox([w_b0, w_theta, w_phi, widgets.HTML("<hr>"), w_b1, w_rf_freq])

# --- Tab 5: Kinetics & Time ---
w_ks = widgets.FloatText(value=0.0, step=0.01, description='k_S (1/µs):', style=style)
w_kt = widgets.FloatText(value=0.0, step=0.01, description='k_T (1/µs):', style=style)
w_tmax = widgets.IntText(value=1000, description='T_MAX (µs):', style=style)
tab_kinetics = widgets.VBox([w_ks, w_kt, w_tmax])

# Assemble the Tabbed Interface
ui_tabs = widgets.Tab(children=[tab_donor, tab_acceptor, tab_ee, tab_fields, tab_kinetics])
ui_tabs.set_title(0, 'Donor Nuclei')
ui_tabs.set_title(1, 'Acceptor Nuclei')
ui_tabs.set_title(2, 'E-E Couplings')
ui_tabs.set_title(3, 'Fields')
ui_tabs.set_title(4, 'Kinetics')

run_button = widgets.Button(description="Run Simulation", button_style='success')
output_plot = widgets.Output()

ui = widgets.VBox([ui_tabs, run_button, output_plot])
display(ui)

# =============================================================================
# 2. EXECUTION LOGIC
# =============================================================================
def parse_vector_string(vec_str):
    return [float(x.strip()) for x in vec_str.split(',')]

def extract_nucleus_data(widget_list):
    spins, tensors, pols = [], [], []
    for w in widget_list:
        if w['spin'].value > 0:
            spins.append(w['spin'].value)
            tensors.append(np.diag([w['ax'].value / GYRO_E, w['ay'].value / GYRO_E, w['az'].value / GYRO_E]))
            pols.append(parse_vector_string(w['pol'].value))
    return spins, tensors, pols

def on_run_clicked(b):
    output_plot.clear_output(wait=True)
    with output_plot:
        print("Extracting dynamic parameters...")
        
        # Parse arrays of arbitrary length from the dynamic UI blocks
        D_SPINS, A_TENSOR_D_LIST, P_D_LIST = extract_nucleus_data(donor_widgets_list)
        A_SPINS, A_TENSOR_A_LIST, P_A_LIST = extract_nucleus_data(acceptor_widgets_list)
        
        D_TENSOR = np.diag([w_dx.value / GYRO_E, w_dy.value / GYRO_E, w_dz.value / GYRO_E])
        J_EX = w_jex.value / GYRO_E
        
        B0, THETA, PHI = w_b0.value, w_theta.value, w_phi.value
        B1_RF_MT, RF_FREQ_MHZ = w_b1.value, w_rf_freq.value
        k_S, k_T, T_MAX = w_ks.value, w_kt.value, w_tmax.value
        
        GYRO_1H_MHZ = 0.04258 * 2 * np.pi 
        steps = max(300, int(T_MAX * 25))
        times = np.linspace(0, T_MAX, steps)
        
        # Construct and evaluate
        sys_rpm = NSpinRPMSystem(d_spins=D_SPINS, a_spins=A_SPINS)
        solver = NPolarizedSolver(sys_rpm)
        
        H_hf = get_n_spin_anisotropic_hyperfine(sys_rpm, A_TENSOR_D_LIST, A_TENSOR_A_LIST, theta=THETA, phi=PHI)
        H_z = get_n_spin_zeeman(sys_rpm, B0, theta=THETA, phi=PHI)
        H_dip = get_n_spin_dipolar(sys_rpm, D_TENSOR, theta=THETA, phi=PHI)
        H_ex = get_n_spin_exchange(sys_rpm, J_EX)
        
        H_0 = H_hf + H_z + H_dip + H_ex
        
        if B1_RF_MT > 0.0 and D_SPINS:
            H_1 = (B1_RF_MT * GYRO_1H_MHZ) * sys_rpm.ID[0]['x']
            H_tot_t = [H_0, [H_1, 'cos(w * t)']]
            args = {'w': RF_FREQ_MHZ * 2 * np.pi}
        else:
            H_tot_t = H_0
            args = {}
            
        c_ops = solver.get_collapse_ops(k_S, k_T)
        pop_ops = solver.get_population_ops()
        e_ops = [pop_ops['S'], pop_ops['Tp'], pop_ops['T0'], pop_ops['Tm']]
        
        rho0 = solver.get_initial_rho(P_D_LIST, P_A_LIST)
        
        print(f"Evaluating Hilbert space over {steps} steps...")
        result = qt.mesolve(H_tot_t, rho0, times, c_ops, e_ops=e_ops, args=args)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, np.real(result.expect[0]), label=r'Singlet ($S$)', color='#1f77b4', linewidth=2)
        ax.plot(times, np.real(result.expect[1]), label=r'Triplet ($T_+$)', color='#d62728', linewidth=1.5, alpha=0.8)
        ax.plot(times, np.real(result.expect[2]), label=r'Triplet ($T_0$)', color='#2ca02c', linewidth=1.5, alpha=0.8)
        ax.plot(times, np.real(result.expect[3]), label=r'Triplet ($T_-$)', color='#ff7f0e', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel(r'Time ($\mu$s)')
        ax.set_ylabel('Population')
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'RPM Dynamics (B0 = {B0} mT)')
        
        plt.tight_layout()
        plt.show()

run_button.on_click(on_run_clicked)
# %%
