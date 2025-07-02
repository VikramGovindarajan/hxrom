import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_points = 500
time = np.linspace(0, 500, n_points)  # Assume 0.5s interval

# Generate synthetic transient input data
T_in_pri = 550 + 10 * np.sin(0.02 * time) # + np.random.normal(0, 1, n_points)
T_in_sec = 355 + 10 * np.cos(0.03 * time) # + np.random.normal(0, 1, n_points)
m_dot_pri = 1500 + 50 * np.sin(0.01 * time) # + np.random.normal(0, 0.05, n_points)
m_dot_sec = 1450 + 50 * np.cos(0.015 * time) # + np.random.normal(0, 0.05, n_points)

# Generate synthetic output data with some lag and nonlinearity
T_out_pri = T_in_pri - 150 + 2 * np.sin(0.05 * time) # + np.random.normal(0, 1.5, n_points)
T_out_sec = T_in_sec + 170 + 3 * np.cos(0.04 * time) # + np.random.normal(0, 1.5, n_points)

# Combine into DataFrame
df = pd.DataFrame({
    'time': time,
    'T_in_pri': T_in_pri,
    'T_in_sec': T_in_sec,
    'm_dot_pri': m_dot_pri,
    'm_dot_sec': m_dot_sec,
    'T_out_pri': T_out_pri,
    'T_out_sec': T_out_sec,
})

# Save to CSV
csv_path = "transient_data.csv"
df.to_csv(csv_path, index=False)

csv_path
