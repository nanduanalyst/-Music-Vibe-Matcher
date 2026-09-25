# 🎵 Music Vibe Matcher

**A content-based music recommendation app that finds tracks matching a "vibe" you describe with sliders (energy, tempo, mood, popularity, loudness) — powered by a cosine-similarity k-Nearest Neighbors model trained on a 2020–2025 Spotify streaming dataset.**

Unlike collaborative-filtering recommenders that need user listening history, this system works purely on **audio characteristics**, so it can recommend tracks based on the *sound* you want, even with zero prior data about you.

---

## 📁 Project Structure

```
music-vibe-matcher/
├── app.py                                    # Streamlit web app (run this to use the tool)
├── train_model.py                            # Trains the KNN model from raw data
├── requirements.txt                          # Python dependencies
├── data/
│   └── spotify_artist_streaming_2020_2025.csv   # Raw dataset (50k tracks, 33 features)
├── models/
│   ├── knn_model.pkl                         # Trained NearestNeighbors model (cosine distance)
│   ├── scaler.pkl                            # StandardScaler fitted on training features
│   ├── feature_cols.pkl                      # Ordered list of feature column names
│   └── tracks_metadata.feather               # Lightweight track lookup table for the UI
├── .gitignore
└── README.md
```

## 🧠 How It Works

1. **`train_model.py`** loads the raw CSV, one-hot encodes `popularity_category` and `loudness_category`, scales 13 numeric/categorical features with `StandardScaler`, and fits a `NearestNeighbors` model (cosine distance, k=5) on the scaled feature matrix. It saves the model, scaler, feature list, and a metadata table to `models/`.
2. **`app.py`** loads those saved artifacts, lets you set desired audio characteristics with sliders in the sidebar, builds a matching feature vector, scales it with the *same* scaler, and asks the KNN model for the closest tracks by cosine distance. Results are shown with a **match score** (derived from distance) plus genre, tempo, and energy.

### Features used by the model
`energy`, `mode` (major/minor), `instrumentalness`, `tempo`, `is_explicit_bool`, `log_stream_count`, `upbeat_score`, `artist_track_count`, plus one-hot encoded `popularity_category` and `loudness_category`.

---

## ▶️ Line-by-Line Execution Guide

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/music-vibe-matcher.git
cd music-vibe-matcher
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the model
The trained artifacts are already included in `models/`, so this step is optional — only run it if you change the dataset or feature engineering.
```bash
python train_model.py
```
This regenerates `knn_model.pkl`, `scaler.pkl`, `feature_cols.pkl`, and `tracks_metadata.feather` inside `models/`.

### 5. Run the Streamlit app locally
```bash
streamlit run app.py
```
Streamlit will start a local server and print a URL — normally:
```
Local URL:   http://localhost:8501
Network URL: http://192.168.x.x:8501
```
Open that link in your browser. Use the sidebar sliders to set the vibe you want, click **🚀 Find Matching Tracks**, and see your recommendations.

---

## 🚀 Deploying to a Public Live URL (Streamlit Community Cloud — free)

I can't host the site for you directly, but here's the exact path to get a live `https://your-app-name.streamlit.app` link in a few minutes, once your code is pushed to GitHub:

1. Push this project to a **public** GitHub repository (see steps below).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, the branch (`main`), and set **Main file path** to `app.py`.
5. Click **"Deploy"**. Streamlit Cloud installs `requirements.txt` automatically and builds the app.
6. In a minute or two, you'll get a live public URL like:
   ```
   https://music-vibe-matcher-<random-id>.streamlit.app
   ```
7. Share that link with anyone — it updates automatically every time you push new commits to `main`.

> **Note:** Because `data/spotify_artist_streaming_2020_2025.csv` (~12 MB) and `models/knn_model.pkl` (~5 MB) are both under GitHub's 100 MB per-file limit, they can be committed normally — no Git LFS needed.

---

## 📦 Pushing This Project to GitHub

```bash
cd music-vibe-matcher
git init
git add .
git commit -m "Initial commit: Music Vibe Matcher"
git branch -M main
git remote add origin https://github.com/<your-username>/music-vibe-matcher.git
git push -u origin main
```

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit** — interactive web UI
- **scikit-learn** — `StandardScaler`, `NearestNeighbors` (cosine distance)
- **pandas / pyarrow** — data handling and fast `.feather` I/O
- **joblib** — model persistence

## 📊 Dataset
`spotify_artist_streaming_2020_2025.csv` — ~50,000 tracks with audio features (energy, tempo, danceability, loudness, instrumentalness, etc.), streaming counts, popularity tiers, and release metadata spanning 2020–2025.

## 🗺️ Possible Improvements
- Add a "search by track name" mode (find tracks similar to a specific song instead of manual sliders)
- Swap KNN for an approximate nearest-neighbor index (e.g. FAISS/Annoy) for larger catalogs
- Add genre/decade filters to narrow the candidate pool before matching
- Add audio previews (Spotify embed) next to each recommendation

## 📄 License
Add a license of your choice (e.g. MIT) before making the repository public if you plan to share or accept contributions.
