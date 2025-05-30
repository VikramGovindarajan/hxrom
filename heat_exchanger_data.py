import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 100

# Inputs: random but realistic
T_in_pri = np.random.uniform(540, 560, n_samples)      # K
T_in_sec = np.random.uniform(345, 355, n_samples)     # K
m_dot_pri = np.random.uniform(1400., 1600.0, n_samples)      # kg/s
m_dot_sec = np.random.uniform(1400., 1500.0, n_samples)     # kg/s

# Constants
Cp = 1260.  # J/kg-K (assumed constant)
efficiency = 0.8771560495698427  # heat exchanger effectiveness

# Energy balance (Q = m*Cp*dT), assume sec side heats up
delta_T_max = T_in_pri - T_in_sec
Q = efficiency * np.minimum(m_dot_pri, m_dot_sec) * Cp * delta_T_max

# Outputs
T_out_sec = T_in_sec + Q / (m_dot_sec * Cp)
T_out_pri = T_in_pri - Q / (m_dot_pri * Cp)

# Add some Gaussian noise
T_out_pri += np.random.normal(0, 1.0, n_samples)
T_out_sec += np.random.normal(0, 1.0, n_samples)
Q += np.random.normal(0, 500E3, n_samples)  # ±500 kW variation

# Create DataFrame
df = pd.DataFrame({
    'T_in_pri': T_in_pri,
    'T_in_sec': T_in_sec,
    'm_dot_pri': m_dot_pri,
    'm_dot_sec': m_dot_sec,
    'T_out_pri': T_out_pri,
    'T_out_sec': T_out_sec,
    'Q': Q
})

# Save to CSV
df.to_csv('heat_exchanger_data.csv', index=False)
print("Synthetic dataset saved as 'heat_exchanger_data.csv'")
