# 🎵 Spotify Music Vibe Matcher

A Machine Learning based **Music Recommendation System** built with Python, Scikit-learn and Streamlit.

The application recommends tracks based on the **audio characteristics and music preferences** selected by the user. Instead of requiring a user's previous listening history, the system finds tracks with similar audio and popularity characteristics using a **k-Nearest Neighbors (KNN)** model with **Cosine Distance**.

---
## 🌐 Live Demo

👉 [🚀 Try the Movie Recommendation System](https://cbmf4vuewyqdkohsrxuwal.streamlit.app)


## 🎧 Project Overview

**Spotify Music Vibe Matcher** allows users to describe the type of music they want using interactive controls such as:

- Energy
- Tempo
- Upbeat Score
- Instrumentalness
- Major / Minor mode
- Explicit / Non-explicit
- Streaming count
- Artist track count
- Popularity category
- Loudness category

The application then processes these preferences and finds the most similar tracks from the Spotify music dataset.

The recommendation engine uses a **content-based approach**, meaning recommendations are generated from track characteristics rather than user listening history.

---

## 🤖 Machine Learning Workflow

```text
Spotify Music Dataset
        ↓
Load Dataset
        ↓
Feature Engineering
        ↓
One-Hot Encoding
        ↓
Select Model Features
        ↓
StandardScaler
        ↓
KNN Model
        ↓
Cosine Distance
        ↓
Find Nearest Tracks
        ↓
Calculate Match Score
        ↓
Display Recommendations
        ↓
Streamlit Web Application
```

---

## 🧠 How the Recommendation System Works

The recommendation system is built using **k-Nearest Neighbors**.

### Step 1 — Load Dataset

The training script reads:

```text
data/spotify_artist_streaming_2020_2025.csv
```

The dataset contains approximately **50,000 tracks** with audio characteristics, streaming information, popularity categories and release-related information.

---

### Step 2 — Encode Categorical Features

The following categorical features are converted into numerical columns using One-Hot Encoding:

```text
popularity_category
loudness_category
```

This allows the machine learning model to process categorical information numerically.

---

### Step 3 — Select Features

The model uses the following features:

```text
energy
mode
instrumentalness
tempo
is_explicit_bool
log_stream_count
upbeat_score
artist_track_count
popularity_category_Low
popularity_category_Medium
popularity_category_Very High
loudness_category_Moderate
loudness_category_Quiet
```

These are the exact feature columns used during model training.

---

### Step 4 — Feature Scaling

Because the features have different numerical ranges, `StandardScaler` is used to standardize them.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[feature_cols]
)
```

The trained scaler is saved so that the same transformation can be applied to user input later.

---

### Step 5 — Train KNN Model

The recommendation engine uses:

```python
NearestNeighbors(
    n_neighbors=5,
    metric="cosine"
)
```

The model searches for tracks with the smallest cosine distance from the user's selected music profile.

---

### Step 6 — Save Model Artifacts

After training, the following files are generated:

```text
knn_model.pkl
scaler.pkl
feature_cols.pkl
tracks_metadata.feather
```

These files contain the trained model, feature scaler, feature ordering and track metadata required by the application.

---

## 🎚️ User Controls

The Streamlit application provides interactive controls for creating a desired music profile.

### Audio Characteristics

```text
Energy
Tempo (BPM)
Upbeat Score
Instrumentalness
Mode
Explicit Track
```

### Popularity & Volume

```text
Log Stream Count
Artist Track Count
Popularity Category
Loudness Category
```

The user can also select the number of recommendations to display.

---

## 🎯 Recommendation Process

When the user clicks:

```text
🚀 Find Matching Track
```

the application:

1. Collects the selected audio characteristics.
2. Converts categorical values into the required numerical features.
3. Creates the input feature vector.
4. Applies the trained `StandardScaler`.
5. Uses the KNN model to find the nearest tracks.
6. Calculates a match score from cosine distance.
7. Displays the recommended tracks.

The recommendation results show:

- Track name
- Artist
- Match score
- Genre
- Tempo
- Energy



---

## 📊 Dataset

The project uses:

```text
spotify_artist_streaming_2020_2025.csv
```

The dataset contains approximately **50,000 tracks** and includes information such as:

- Track name
- Artist name
- Genre
- Energy
- Tempo
- Danceability
- Loudness
- Instrumentalness
- Streaming count
- Popularity category
- Release information

The model specifically uses a subset of these features for recommendation.

---

## 📁 Project Structure

```text
music-vibe-matcher/
│
├── index.py
│   └── Streamlit application
│
├── train_model.py
│   └── Machine Learning model training script
│
├── data/
│   └── spotify_artist_streaming_2020_2025.csv
│       └── Spotify music dataset
│
├── models/
│   ├── knn_model.pkl
│   │   └── Trained KNN model
│   │
│   ├── scaler.pkl
│   │   └── StandardScaler
│   │
│   ├── feature_cols.pkl
│   │   └── Model feature names
│   │
│   └── tracks_metadata.feather
│       └── Track information used by the app
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Git ignored files
│
└── README.md
    └── Project documentation
```

---

# ▶️ How to Run the Project

## 1. Open the Project Folder

Open a terminal inside the project directory:

```bash
cd music-vibe-matcher
```

---

## 2. Create a Virtual Environment

Creating a virtual environment is recommended.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

The project requires:

```text
Streamlit
Pandas
NumPy
Scikit-learn
Joblib
PyArrow
```



---

## 4. Train the Machine Learning Model

Before running the application, train the model using:

```bash
python train_model.py
```

The training script:

```text
Load Dataset
      ↓
Feature Engineering
      ↓
One-Hot Encoding
      ↓
Feature Selection
      ↓
StandardScaler
      ↓
KNN Training
      ↓
Save Model Files
```

The generated model files are saved inside the `models/` directory.

---

## 5. Run the Streamlit Application

Start the application using:

```bash
streamlit run index.py
```

The application will open in your browser.

Normally Streamlit provides a local address such as:

```text
http://localhost:8501
```

---

## 6. Use the Music Matcher

Once the application opens:

### Step 1
Use the sidebar to select your desired:

```text
Energy
Tempo
Upbeat Score
Instrumentalness
Mode
```

### Step 2

Select additional preferences:

```text
Explicit Track
Popularity Category
Loudness Category
```

### Step 3

Choose how many recommendations you want.

### Step 4

Click:

```text
🚀 Find Matching Track
```

### Step 5

The application displays the closest matching tracks.

Example:

```text
🎯 Top Track Matches

#1
Track Name
Artist Name

Match Score: 92.45%

Genre: Pop
Tempo: 120 BPM
Energy: 0.78
```

---

# 🧩 Main Python Files

## `train_model.py`

This file is responsible for **training the recommendation model**.

It:

- Loads the Spotify dataset
- Encodes categorical variables
- Selects model features
- Scales the data
- Trains the KNN model
- Saves the trained model
- Saves the scaler
- Saves feature names
- Creates the track metadata file

The script uses `NearestNeighbors` with cosine distance.

---

## `index.py`

This is the **Streamlit application**.

It:

- Loads the trained model
- Loads the scaler
- Loads feature information
- Loads track metadata
- Creates the user interface
- Accepts music preferences
- Finds nearest tracks
- Calculates match scores
- Displays recommendations



---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| StandardScaler | Feature scaling |
| NearestNeighbors | Recommendation model |
| Joblib | Model saving/loading |
| PyArrow | Feather file handling |
| Streamlit | Web application |

---

# 📌 Machine Learning Algorithm

### K-Nearest Neighbors

The project uses:

```python
NearestNeighbors(
    n_neighbors=5,
    metric="cosine"
)
```

Cosine distance is used to measure how close the user's selected music profile is to tracks in the dataset.

The smaller the cosine distance, the closer the track is to the requested feature profile.

The application converts this distance into a displayed match score:

```python
match_confidence = (1 - dist) * 100
```



---

# 🔄 Complete Execution Flow

```text
START
  │
  ▼
Load Spotify Dataset
  │
  ▼
Prepare Features
  │
  ▼
Encode Categorical Features
  │
  ▼
Standardize Features
  │
  ▼
Train KNN Model
  │
  ▼
Save Model Artifacts
  │
  ▼
Start Streamlit
  │
  ▼
User Selects Music Preferences
  │
  ▼
Create Input Feature Vector
  │
  ▼
Apply StandardScaler
  │
  ▼
Find Nearest Tracks
  │
  ▼
Calculate Match Score
  │
  ▼
Display Recommended Tracks
  │
  ▼
END
```

---

# 🚀 Future Improvements

Possible improvements for the project include:

- Search by track name
- Recommend tracks similar to a selected song
- Add genre filters
- Add decade filters
- Add Spotify previews
- Add album artwork
- Add more audio features
- Improve the Streamlit interface
- Use approximate nearest-neighbor search for larger datasets

Some of these improvements are also identified in the original project documentation.

---

# 👨‍💻 Author

**Nandu Rastogi**

Aspiring Data Analyst | Machine Learning Engineer

---

## ⭐ Project Summary

**Spotify Music Vibe Matcher** is a Machine Learning powered recommendation system that matches users with tracks based on their desired musical characteristics.

The project demonstrates a complete Machine Learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Feature Scaling
 ↓
KNN Model
 ↓
Cosine Similarity
 ↓
Recommendation
 ↓
Streamlit Application
```

It combines **Data Analysis, Machine Learning, Recommendation Systems and Streamlit** into one practical project.
