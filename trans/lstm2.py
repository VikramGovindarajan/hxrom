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

        # Clamp to physical bounds
        T_in_pri = np.clip(T_in_pri, 200, 600)
        T_in_sec = np.clip(T_in_sec, 200, 600)
        m_dot_pri = np.clip(m_dot_pri, 50, 1600)
        m_dot_sec = np.clip(m_dot_sec, 50, 1500)

        # Physics model
        delta_T = T_in_pri - T_in_sec
        Q = eff * min(m_dot_pri, m_dot_sec) * Cp * delta_T
        T_out_sec_ss = T_in_sec + Q / (m_dot_sec * Cp)
        T_out_pri_ss = T_in_pri - Q / (m_dot_pri * Cp)

        # Transient (first-order lag)
        T_out_pri += alpha * (T_out_pri_ss - T_out_pri)
        T_out_sec += alpha * (T_out_sec_ss - T_out_sec)

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



## Prepare Data for LSTM
from sklearn.preprocessing import MinMaxScaler

# Flatten and normalize
full_data = pd.concat(sequences, ignore_index=True)

input_cols = ['T_in_pri', 'T_in_sec', 'm_dot_pri', 'm_dot_sec']
output_cols = ['T_out_pri', 'T_out_sec']

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_all = scaler_x.fit_transform(full_data[input_cols])
Y_all = scaler_y.fit_transform(full_data[output_cols])

# Rebuild sequences: shape = (num_seq, time_steps, features)
n_seq = len(sequences)
time_steps = len(sequences[0])

X = X_all.reshape(n_seq, time_steps, len(input_cols))
Y = Y_all.reshape(n_seq, time_steps, len(output_cols))

# Split
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)



## Build and Train LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(64, input_shape=(time_steps, len(input_cols)), return_sequences=True),
    Dense(32, activation='relu'),
    Dense(len(output_cols))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    epochs=100, batch_size=16, callbacks=[early_stop])

## Predict Transient Behaviour
# Pick one sequence to test
X_test = X_val[0:1]  # shape: (1, time_steps, 4)
Y_pred = model.predict(X_test)

# Inverse transform to get back to real units
Y_true = scaler_y.inverse_transform(Y_val[0])
Y_pred_rescaled = scaler_y.inverse_transform(Y_pred[0])

import matplotlib.pyplot as plt
plt.plot(Y_true[:, 0], label='True T_out_pri')
plt.plot(Y_pred_rescaled[:, 0], label='Predicted T_out_pri')
plt.plot(Y_true[:, 1], label='True T_out_sec')
plt.plot(Y_pred_rescaled[:, 1], label='Predicted T_out_sec')
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Temperature")
plt.title("LSTM Prediction vs Physics Model")
plt.show()


import pandas as pd
import numpy as np

# Load the new input sequence
X_new_df = pd.read_csv('predict_trans.csv')

# Check column order
required_columns = ['T_in_pri', 'T_in_sec', 'm_dot_pri', 'm_dot_sec']
assert all(col in X_new_df.columns for col in required_columns), "Missing columns in CSV"

# Normalize using the scaler from training
X_new_scaled = scaler_x.transform(X_new_df[required_columns])  # scaler_x from training

# Reshape for LSTM: (1, time_steps, num_features)
X_new_seq = X_new_scaled.reshape(1, X_new_scaled.shape[0], X_new_scaled.shape[1])

# Predict
Y_new_pred_scaled = model.predict(X_new_seq)  # shape: (1, time_steps, 2)

# Inverse transform to get actual temperatures
Y_new_pred = scaler_y.inverse_transform(Y_new_pred_scaled[0])  # shape: (time_steps, 2)

# Save predicted outputs
Y_new_df = pd.DataFrame(Y_new_pred, columns=['T_out_pri', 'T_out_sec'])
Y_new_df.to_csv('Y_new_pred.csv', index=False)

import matplotlib.pyplot as plt

# Time steps
time = np.arange(Y_new_pred.shape[0])  # assuming shape = (time_steps, 2)

# Extract predictions
T_out_pri = Y_new_pred[:, 0]
T_out_sec = Y_new_pred[:, 1]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, T_out_pri, label='Predicted T_out_pri', color='tab:red')
plt.plot(time, T_out_sec, label='Predicted T_out_sec', color='tab:blue')
plt.xlabel('Time step')
plt.ylabel('Temperature (°C)')
plt.title('Predicted Outlet Temperatures from LSTM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
