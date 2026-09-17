import subprocess
import sys

# Define the exact sequence of generation and plotting scripts
scripts = [
    "/home/tristengwynn/nuclear-polarisation/Figure_5/5_generate_mult_nuc.py",
    "/home/tristengwynn/nuclear-polarisation/Figure_5/5_plot_mult_nuc.py",
    "/home/tristengwynn/nuclear-polarisation/Figure_6/6_generate_rate_check.py",
    "/home/tristengwynn/nuclear-polarisation/Figure_6/6_plot_rate_check.py",
    "/home/tristengwynn/nuclear-polarisation/Figure_7/7_generate_ee_data.py",
    "/home/tristengwynn/nuclear-polarisation/Figure_7/7_plot_ee.py"
]

for script in scripts:
    print(f"--- Starting {script} ---")
    
    # sys.executable ensures the active virtual environment is used
    result = subprocess.run([sys.executable, script])
    
    # Halt the entire pipeline if any script fails
    if result.returncode != 0:
        print(f"CRITICAL: {script} failed with exit code {result.returncode}. Halting pipeline.")
        sys.exit(result.returncode)
        
    print(f"--- {script} completed successfully ---\n")