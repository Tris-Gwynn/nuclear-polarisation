# ciss_rpm/constants.py
import numpy as np
# Gyromagnetic ratio conversion factor (mT to frequency units)
# Assumed 28 MHz/mT based on your previous code
GYRO_E = 28 * 2 * np.pi
GYRO_H1 = 0.04258 * 2 * np.pi
GYRO_N14 = 0.003077 * 2 * np.pi
# Simulation settings
DEFAULT_STEPS = 250
RESOLUTION = 36
SWEEP_STEPS = 31
