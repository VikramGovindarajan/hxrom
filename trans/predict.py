import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the scaler
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load the saved model
model = load_model('lstm_model.keras')

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
