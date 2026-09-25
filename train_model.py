"""
train_model.py
---------------
Trains the k-Nearest Neighbors "vibe matcher" model for Music Vibe Matcher.

It reads the raw Spotify streaming dataset, engineers the same features the
Streamlit app expects, fits a StandardScaler + cosine-distance KNN model,
and saves all artifacts needed by app.py into the models/ folder.

Run this once before starting the app (or any time the dataset changes):

    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "spotify_artist_streaming_2020_2025.csv")
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load dataset
print(f"Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Nulls in track_name:", df["track_name"].isnull().sum())
print("Unique track names:", df["track_name"].nunique())

# 2. One-Hot Encode categorical features
df = pd.get_dummies(
    df, columns=["popularity_category", "loudness_category"], drop_first=False
)

# 3. Define the exact feature columns used by the model
feature_cols = [
    "energy",
    "mode",
    "instrumentalness",
    "tempo",
    "is_explicit_bool",
    "log_stream_count",
    "upbeat_score",
    "artist_track_count",
    "popularity_category_Low",
    "popularity_category_Medium",
    "popularity_category_Very High",
    "loudness_category_Moderate",
    "loudness_category_Quiet",
]

# Ensure boolean fields are numeric (0 or 1)
df["is_explicit_bool"] = df["is_explicit_bool"].astype(int)
for col in [
    "popularity_category_Low",
    "popularity_category_Medium",
    "popularity_category_Very High",
    "loudness_category_Moderate",
    "loudness_category_Quiet",
]:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].astype(int)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# 5. Fit k-NN model with Cosine Distance
knn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
knn_model.fit(X_scaled)

# 6. Save artifacts
joblib.dump(knn_model, os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

# Save a lightweight metadata reference for fast retrieval in the Streamlit app
metadata_cols = [
    "track_name",
    "artist_name",
    "genre",
    "tempo",
    "energy",
    "stream_count",
]
df[metadata_cols].reset_index(drop=True).to_feather(
    os.path.join(MODELS_DIR, "tracks_metadata.feather")
)

print("Training complete! Model artifacts saved to the models/ folder.")
