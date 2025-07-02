import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('transient_data.csv')

inputs = ['T_in_hot', 'T_in_cold', 'm_dot_hot', 'm_dot_cold']
outputs = ['T_out_hot', 'T_out_cold', 'Q']

X = df[inputs].values
Y = df[outputs].values

# Normalize
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

# Sequence generation (window size = 10)
def create_sequences(X, Y, window=10):
    X_seq, Y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        Y_seq.append(Y[i+window])
    return np.array(X_seq), np.array(Y_seq)

X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, window=10)

# Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
Y_train, Y_test = Y_seq[:split], Y_seq[split:]

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(Y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=1)

# Predict
Y_pred_scaled = model.predict(X_test)
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
Y_true = y_scaler.inverse_transform(Y_test)

# Evaluate
mse = mean_squared_error(Y_true, Y_pred)
print(f"MSE: {mse:.2f}")

# Plot comparison
for i, col in enumerate(outputs):
    plt.figure(figsize=(6, 4))
    plt.plot(Y_true[:, i], label='Actual')
    plt.plot(Y_pred[:, i], label='Predicted', linestyle='--')
    plt.title(f'{col} (Actual vs Predicted)')
    plt.xlabel('Time step')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
