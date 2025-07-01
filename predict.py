import pandas as pd
import joblib

# Load model and scalers
model = joblib.load('mlp_model.pkl')
X_scaler = joblib.load('x_scaler.pkl')
Y_scaler = joblib.load('y_scaler.pkl')

# Load new input data from CSV
new_X = pd.read_csv('new_input_data.csv')

# Ensure the column order matches the training features
new_X = new_X[['T_in_pri', 'T_in_sec', 'm_dot_pri', 'm_dot_sec']]

# Scale the input using the previously fitted scaler
new_X_scaled = X_scaler.transform(new_X)

# Predict scaled output
new_Y_scaled = model.predict(new_X_scaled)

# Inverse transform to get actual predictions
new_Y = Y_scaler.inverse_transform(new_Y_scaled)

# Convert predictions to DataFrame for easy handling
predictions_df = pd.DataFrame(new_Y, columns=['T_out_pri', 'T_out_sec', 'Q'])

# Save predictions to CSV if needed
predictions_df.to_csv('predicted_outputs.csv', index=False)

# Optionally print predictions
print(predictions_df)
