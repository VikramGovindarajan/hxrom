import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# === Load model and scalers ===
model = load_model('lstm_model.keras')
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# === Load new input sequence ===
X_new_df = pd.read_csv('predict_trans.csv')
required_columns = ['T_in_pri', 'T_in_sec', 'm_dot_pri', 'm_dot_sec']
assert all(col in X_new_df.columns for col in required_columns), "Missing input columns"

# === Inference Parameters ===
T_out_prev = [384.656085, 526.0454296661194]  # nominal initial outputs
X_seq_scaled = []
predictions = []

# === Step-by-step prediction using model ===
for i in range(len(X_new_df)):
    row = X_new_df.iloc[i]

    # Build full input including T_out_prev
    x_full = [
        row['T_in_pri'], row['T_in_sec'],
        row['m_dot_pri'], row['m_dot_sec'],
        T_out_prev[0], T_out_prev[1]
    ]

    x_scaled = scaler_x.transform([x_full]).reshape(1, 1, -1)
    y_scaled = model.predict(x_scaled, verbose=0)  # shape (1, 1, 2)
    y = scaler_y.inverse_transform(y_scaled.reshape(1, -1))[0]  # fix here

    predictions.append(y)
    T_out_prev = y  # Feed prediction into next step

# === Convert predictions to DataFrame ===
Y_new_pred = np.array(predictions)
Y_new_df = pd.DataFrame(Y_new_pred, columns=['T_out_pri', 'T_out_sec'])
Y_new_df.to_csv('Y_new_pred.csv', index=False)

# === Plot ===
time = np.arange(len(Y_new_df))
plt.figure(figsize=(10, 5))
plt.plot(time, Y_new_df['T_out_pri'], label='Predicted T_out_pri', color='tab:red')
plt.plot(time, Y_new_df['T_out_sec'], label='Predicted T_out_sec', color='tab:blue')
plt.xlabel('Time step')
plt.ylabel('Temperature (°C)')
plt.title('Predicted Outlet Temperatures from LSTM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
