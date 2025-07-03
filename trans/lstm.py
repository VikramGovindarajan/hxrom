#lstm.py
import sys
import numpy as np
import pandas as pd

import pickle

# Load the list of DataFrames
with open('sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

## Prepare Data for LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Flatten all sequences for global scaling
full_data = pd.concat(sequences, ignore_index=True)

input_cols_base = ['T_in_pri', 'T_in_sec', 'm_dot_pri', 'm_dot_sec']
output_cols = ['T_out_pri', 'T_out_sec']

# Initialize scalers
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit scalers
scaler_x.fit(full_data[input_cols_base + output_cols])  # Include T_out_* for prev input
scaler_y.fit(full_data[output_cols])

# Prepare X, Y with T_out_prev included
X_list = []
Y_list = []

for df in sequences:
    X_seq = []
    Y_seq = []

    # Initial previous outlet temperature = nominal
    T_out_prev = [384.656085, 526.0454296661194]

    for i in range(len(df)):
        row = df.iloc[i]

        X_row = [
            row['T_in_pri'], row['T_in_sec'],
            row['m_dot_pri'], row['m_dot_sec'],
            T_out_prev[0], T_out_prev[1]
        ]
        Y_row = [row['T_out_pri'], row['T_out_sec']]

        # Update for next time step
        T_out_prev = Y_row

        # Scale and store
        X_scaled = scaler_x.transform([X_row])[0]
        Y_scaled = scaler_y.transform([Y_row])[0]

        X_seq.append(X_scaled)
        Y_seq.append(Y_scaled)

    X_list.append(X_seq)
    Y_list.append(Y_seq)

# Convert to numpy arrays
X = np.array(X_list)  # shape: (n_seq, time_steps, 6)
Y = np.array(Y_list)  # shape: (n_seq, time_steps, 2)

# Train/val split
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)



## Build and Train LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(64, input_shape=(100, 6), return_sequences=True),
    Dense(32, activation='relu'),
    Dense(2)
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

# Save the model
model.save('lstm_model.keras')  # You can use .keras extension as well

import joblib

# After fitting the scaler
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')  # If you also need to scale outputs
