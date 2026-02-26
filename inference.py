import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Constants
T = 200
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

# Load model
model = tf.keras.models.load_model("model/airdraw_model.keras")

# Load StandardScaler instead of mean/std
scaler = joblib.load("model/standard_scaler.pkl")

def resample_to_T(df, T=200):
    old_len = len(df)
    old_idx = np.linspace(0, old_len - 1, old_len)
    new_idx = np.linspace(0, old_len - 1, T)

    resampled = np.zeros((T, len(FEATURES)))
    for i, col in enumerate(FEATURES):
        resampled[:, i] = np.interp(new_idx, old_idx, df[col].values)

    return resampled

def predict_digit(csv_path):
    df = pd.read_csv(csv_path)

    X = resample_to_T(df)          # (200, 6)

    # reshape to 2D for scaler
    X_2d = X.reshape(-1, len(FEATURES))

    # apply scaler
    X_scaled = scaler.transform(X_2d)

    # reshape back to 3D
    X_scaled = X_scaled.reshape(1, T, len(FEATURES))

    probs = model.predict(X_scaled)[0]
    pred = np.argmax(probs)

    return pred, probs

# Example usage
# Take CSV file path from user
file_path = input("Enter the path of the digit CSV file: ")

digit, probs = predict_digit(file_path)

print("Predicted digit:", digit)
print("Probabilities:", probs)