"""
app.py
------
Music Vibe Matcher — a Streamlit app that recommends tracks whose audio
characteristics best match a set of sliders you choose (energy, tempo,
mood, popularity tier, etc.), using a cosine-distance k-Nearest Neighbors
model trained on a 2020-2025 Spotify streaming dataset.

Run with:
    streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Music Vibe Matcher", page_icon="🎵", layout="wide"
)

MODELS_DIR = "models"


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    metadata = pd.read_feather(os.path.join(MODELS_DIR, "tracks_metadata.feather"))
    return model, scaler, feature_cols, metadata


model, scaler, feature_cols, metadata = load_artifacts()

st.title("🎵 Music Vibe Matcher")
st.markdown(
    "Set your desired audio characteristics below to find the best matching "
    "**tracks** from the catalog."
)

# ---------------------------------------------------------------- Input UI
with st.sidebar:
    st.header("🎚️ Audio Characteristics")

    energy = st.slider("Energy", 0.0, 1.0, 0.65, 0.01)
    tempo = st.slider("Tempo (BPM)", 60.0, 220.0, 120.0, 1.0)
    upbeat_score = st.slider("Upbeat Score", 0.0, 1.0, 0.60, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.05, 0.01)
    mode = st.radio(
        "Mode", options=[1, 0], format_func=lambda x: "Major" if x == 1 else "Minor"
    )
    is_explicit = st.checkbox("Explicit Track")

    st.header("📈 Popularity & Volume")
    log_stream_count = st.slider("Log Stream Count", 4.0, 18.0, 10.0, 0.1)
    artist_track_count = st.number_input(
        "Artist Track Count", min_value=1, max_value=15000, value=2500
    )

    pop_tier = st.selectbox(
        "Popularity Category", ["Low", "Medium", "High", "Very High"]
    )
    loud_tier = st.selectbox("Loudness Category", ["Moderate", "Quiet", "Loud"])

    top_k = st.slider("Number of recommendations", 1, 10, 5)

# ------------------------------------------------ Build the input feature vector
pop_low = 1 if pop_tier == "Low" else 0
pop_med = 1 if pop_tier == "Medium" else 0
pop_vhigh = 1 if pop_tier == "Very High" else 0

loud_mod = 1 if loud_tier == "Moderate" else 0
loud_quiet = 1 if loud_tier == "Quiet" else 0

input_dict = {
    "energy": energy,
    "mode": mode,
    "instrumentalness": instrumentalness,
    "tempo": tempo,
    "is_explicit_bool": int(is_explicit),
    "log_stream_count": log_stream_count,
    "upbeat_score": upbeat_score,
    "artist_track_count": artist_track_count,
    "popularity_category_Low": pop_low,
    "popularity_category_Medium": pop_med,
    "popularity_category_Very High": pop_vhigh,
    "loudness_category_Moderate": loud_mod,
    "loudness_category_Quiet": loud_quiet,
}

input_df = pd.DataFrame([input_dict])[feature_cols]

# --------------------------------------------------------------- Recommend
if st.button("🚀 Find Matching Tracks", use_container_width=True):
    scaled_input = scaler.transform(input_df)
    distances, indices = model.kneighbors(scaled_input, n_neighbors=top_k)

    st.subheader("🎯 Top Track Matches")

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        track_info = metadata.iloc[idx]
        match_confidence = round((1 - dist) * 100, 2)

        with st.container():
            cols = st.columns([1, 4, 2, 2])
            cols[0].metric("Rank", f"#{rank}")
            cols[1].markdown(
                f"### {track_info['track_name']}\n**Artist:** {track_info['artist_name']}"
            )
            cols[2].metric("Match Score", f"{match_confidence}%")
            cols[3].caption(
                f"Genre: {track_info['genre']}\n\n"
                f"Tempo: {track_info['tempo']:.0f} BPM | Energy: {track_info['energy']:.2f}"
            )
            st.divider()
