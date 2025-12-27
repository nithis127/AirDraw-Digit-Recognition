import numpy as np
import pandas as pd
import tensorflow as tf

# Constants
T = 200
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

# Load model
model = tf.keras.models.load_model("model/airdraw_model.keras")

# Load normalization stats
norm = np.load("model/norm_stats.npz")
mean = norm["mean"]
std = norm["std"]

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

    X = resample_to_T(df)
    X = (X - mean) / std
    X = X[np.newaxis, :, :]  # shape (1, 200, 6)

    probs = model.predict(X)[0]
    pred = np.argmax(probs)

    return pred, probs

# Example usage
# Take CSV file path from user
file_path = input("Enter the path of the digit CSV file: ")

digit, probs = predict_digit(file_path)

print("Predicted digit:", digit)
print("Probabilities:", probs)