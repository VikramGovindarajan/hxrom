import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_points = 500
time = np.linspace(0, 250, n_points)  # Assume 0.5s interval

# Generate synthetic transient input data
T_in_hot = 800 + 20 * np.sin(0.02 * time) + np.random.normal(0, 1, n_points)
T_in_cold = 300 + 10 * np.cos(0.03 * time) + np.random.normal(0, 1, n_points)
m_dot_hot = 5 + 0.5 * np.sin(0.01 * time) + np.random.normal(0, 0.05, n_points)
m_dot_cold = 6 + 0.5 * np.cos(0.015 * time) + np.random.normal(0, 0.05, n_points)

# Generate synthetic output data with some lag and nonlinearity
T_out_hot = T_in_hot - 30 + 2 * np.sin(0.05 * time) + np.random.normal(0, 1.5, n_points)
T_out_cold = T_in_cold + 80 + 3 * np.cos(0.04 * time) + np.random.normal(0, 1.5, n_points)
Q = m_dot_hot * (T_in_hot - T_out_hot) * 4.18  # Simple Q = m*c*ΔT (approximation)

# Combine into DataFrame
df = pd.DataFrame({
    'time': time,
    'T_in_hot': T_in_hot,
    'T_in_cold': T_in_cold,
    'm_dot_hot': m_dot_hot,
    'm_dot_cold': m_dot_cold,
    'T_out_hot': T_out_hot,
    'T_out_cold': T_out_cold,
    'Q': Q
})

# Save to CSV
csv_path = "/mnt/data/transient_data.csv"
df.to_csv(csv_path, index=False)

csv_path
