import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('heat_exchanger_data.csv')
X = df[['T_in_hot', 'T_in_cold', 'm_dot_hot', 'm_dot_cold']]
Y = df[['T_out_hot', 'T_out_cold', 'Q']]

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = MLPRegressor(hidden_layer_sizes=(128, 128),
                     activation='relu',
                     solver='adam',
                     learning_rate_init=0.001,
                     max_iter=10000,
                     random_state=42)

# Scale input features
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Scale output features
Y_scaler = StandardScaler()
Y_train_scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

model.fit(X_train_scaled, Y_train_scaled)

# Predict and inverse-transform predictions
Y_pred_scaled = model.predict(X_test_scaled)
Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)

mse = mean_squared_error(Y_test, Y_pred)
print(f"MSE: {mse}")

import matplotlib.pyplot as plt

# Convert predicted and true outputs to DataFrame for easy plotting
Y_test_df = pd.DataFrame(Y_test, columns=['T_out_hot', 'T_out_cold', 'Q'], index=X_test.index)
Y_pred_df = pd.DataFrame(Y_pred, columns=['T_out_hot', 'T_out_cold', 'Q'], index=X_test.index)

# Plot predicted vs actual for each output
for col in Y_test.columns:
    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test_df[col], Y_pred_df[col], alpha=0.7, edgecolors='k')
    plt.plot([Y_test_df[col].min(), Y_test_df[col].max()],
             [Y_test_df[col].min(), Y_test_df[col].max()],
             'r--', lw=2)
    plt.xlabel(f'Actual {col}')
    plt.ylabel(f'Predicted {col}')
    plt.title(f'{col}: Predicted vs Actual')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
