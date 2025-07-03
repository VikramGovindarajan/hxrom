import sys
import numpy as np
import pandas as pd

import pickle

# Load the list of DataFrames
with open('sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

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

# Save the model
model.save('lstm_model.keras')  # You can use .keras extension as well

import joblib

# After fitting the scaler
joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')  # If you also need to scale outputs
