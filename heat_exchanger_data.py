import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 100

# Inputs: random but realistic
T_in_hot = np.random.uniform(400, 500, n_samples)      # K
T_in_cold = np.random.uniform(280, 320, n_samples)     # K
m_dot_hot = np.random.uniform(1.5, 3.0, n_samples)      # kg/s
m_dot_cold = np.random.uniform(1.5, 3.0, n_samples)     # kg/s

# Constants
Cp = 1000  # J/kg-K (assumed constant)
efficiency = 0.85  # heat exchanger effectiveness

# Energy balance (Q = m*Cp*dT), assume cold side heats up
delta_T_max = T_in_hot - T_in_cold
Q = efficiency * np.minimum(m_dot_hot, m_dot_cold) * Cp * delta_T_max

# Outputs
T_out_cold = T_in_cold + Q / (m_dot_cold * Cp)
T_out_hot = T_in_hot - Q / (m_dot_hot * Cp)

# Add some Gaussian noise
T_out_hot += np.random.normal(0, 1.0, n_samples)
T_out_cold += np.random.normal(0, 1.0, n_samples)
Q += np.random.normal(0, 500, n_samples)  # ±500 W variation

# Create DataFrame
df = pd.DataFrame({
    'T_in_hot': T_in_hot,
    'T_in_cold': T_in_cold,
    'm_dot_hot': m_dot_hot,
    'm_dot_cold': m_dot_cold,
    'T_out_hot': T_out_hot,
    'T_out_cold': T_out_cold,
    'Q': Q
})

# Save to CSV
df.to_csv('heat_exchanger_data.csv', index=False)
print("Synthetic dataset saved as 'heat_exchanger_data.csv'")
