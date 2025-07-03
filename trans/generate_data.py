#generate_data.py
import sys
import numpy as np
import pandas as pd

## Generate Data

def simulate_physics_model_from_nominal(perturb_funcs, steps=100):
    data = []

    # Constants
    Cp = 1260.0  # J/kg-K
    eff = 0.8771560495698427
    alpha = 0.3  # inertia weight

    # Nominal input conditions
    T_in_pri_nom = 550.0
    T_in_sec_nom = 355.0
    m_dot_pri_nom = 1500.0
    m_dot_sec_nom = 1450.0

    # Steady-state outputs (from effectiveness method)
    delta_T_max = T_in_pri_nom - T_in_sec_nom
    Q_nom = eff * min(m_dot_pri_nom, m_dot_sec_nom) * Cp * delta_T_max
    T_out_sec = T_in_sec_nom + Q_nom / (m_dot_sec_nom * Cp)
    T_out_pri = T_in_pri_nom - Q_nom / (m_dot_pri_nom * Cp)

    for t in range(steps):
        # Apply perturbations around nominal inputs
        T_in_pri = T_in_pri_nom + perturb_funcs[0](t)
        T_in_sec = T_in_sec_nom + perturb_funcs[1](t)
        m_dot_pri = m_dot_pri_nom + perturb_funcs[2](t)
        m_dot_sec = m_dot_sec_nom + perturb_funcs[3](t)

        # # Clamp to physical bounds
        # T_in_pri = np.clip(T_in_pri, 200, 600)
        # T_in_sec = np.clip(T_in_sec, 200, 600)
        # m_dot_pri = np.clip(m_dot_pri, 50, 1600)
        # m_dot_sec = np.clip(m_dot_sec, 50, 1500)

        # Physics model
        delta_T = T_in_pri - T_in_sec
        Q = eff * min(m_dot_pri, m_dot_sec) * Cp * delta_T
        T_out_sec_nom = T_in_sec + Q / (m_dot_sec * Cp)
        T_out_pri_nom = T_in_pri - Q / (m_dot_pri * Cp)

        # Transient (first-order lag)
        T_out_pri += alpha * (T_out_pri_nom - T_out_pri)
        T_out_sec += alpha * (T_out_sec_nom - T_out_sec)

        data.append([
            T_in_pri, T_in_sec, m_dot_pri, m_dot_sec,
            T_out_pri, T_out_sec
        ])

    return pd.DataFrame(data, columns=[
        'T_in_pri', 'T_in_sec', 'm_dot_pri', 'm_dot_sec',
        'T_out_pri', 'T_out_sec'
    ])

def random_perturbation_generator(input_type):
    if input_type == "temperature":
        A = np.random.uniform(-200, 200)
    elif input_type == "flow":
        A = np.random.uniform(-1000, 1000)
    else:
        raise ValueError("Invalid input type")

    f = np.random.uniform(0.01, 0.1)  # frequency
    phase = 0. #np.random.uniform(0, 2 * np.pi)

    return lambda t: A * np.sin(f * t + phase)


sequences = []
for _ in range(100):
    perturb_funcs = [
        random_perturbation_generator("temperature"),  # T_in_pri
        random_perturbation_generator("temperature"),  # T_in_sec
        random_perturbation_generator("flow"),         # m_dot_pri
        random_perturbation_generator("flow")          # m_dot_sec
    ]
    df = simulate_physics_model_from_nominal(perturb_funcs, steps=100)
    sequences.append(df)

# Add steady-state sequences (no perturbation)
perturb_funcs_steady = [lambda t: 0.0] * 4
for _ in range(20):
    df = simulate_physics_model_from_nominal([lambda t: 0.0]*4, steps=100)
    sequences.append(df)

import pickle

# Save list of DataFrames to a .pkl file
with open('sequences.pkl', 'wb') as f:
    pickle.dump(sequences, f)

