from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import warnings
import os
import urllib.request
from flask_cors import CORS
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder=".", static_folder=".")
CORS(app)

# Download model files if they don't exist
MODEL_FILES = {
    "Crop_Recom.pkl": "https://github.com/Ttejas09/CropiAI/releases/download/v1.0/Crop_Recom.pkl",
    "dist_crop_season.pkl": "https://github.com/Ttejas09/CropiAI/releases/download/v1.0/dist_crop_season.pkl",
    "crop_predict.pkl": "https://github.com/Ttejas09/CropiAI/releases/download/v1.0/crop_predict.pkl"
}

def download_model(filename, url):
    """Download model file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise

# Lazy load models - only download when first needed
crop_recom_model = None
dist_crop_season_transformer = None
crop_predict_model = None

def load_crop_recom_model():
    global crop_recom_model
    if crop_recom_model is None:
        download_model("Crop_Recom.pkl", MODEL_FILES["Crop_Recom.pkl"])
        with open("Crop_Recom.pkl", "rb") as f:
            crop_recom_model = pickle.load(f)
    return crop_recom_model

def load_crop_predict_model():
    global crop_predict_model
    if crop_predict_model is None:
        download_model("crop_predict.pkl", MODEL_FILES["crop_predict.pkl"])
        with open("crop_predict.pkl", "rb") as f:
            crop_predict_model = pickle.load(f)
    return crop_predict_model

def load_dist_crop_season_model():
    global dist_crop_season_transformer
    if dist_crop_season_transformer is None:
        download_model("dist_crop_season.pkl", MODEL_FILES["dist_crop_season.pkl"])
        with open("dist_crop_season.pkl", "rb") as f:
            dist_crop_season_transformer = pickle.load(f)
    return dist_crop_season_transformer

# Pre-load CROP_LABELS and feature names (these are stored in the models)
CROP_LABELS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

CROP_EMOJIS = {
    'apple': '🍎', 'banana': '🍌', 'blackgram': '🫘', 'chickpea': '🫛',
    'coconut': '🥥', 'coffee': '☕', 'cotton': '🌿', 'grapes': '🍇',
    'jute': '🌾', 'kidneybeans': '🫘', 'lentil': '🫘', 'maize': '🌽',
    'mango': '🥭', 'mothbeans': '🫘', 'mungbean': '🌱', 'muskmelon': '🍈',
    'orange': '🍊', 'papaya': '🧡', 'pigeonpeas': '🫘', 'pomegranate': '🍎',
    'rice': '🌾', 'watermelon': '🍉'
}


PREDICT_FEATURES = None
DISTRICTS = None
CROPS_PREDICT = None
SEASONS = None

def get_predict_features():
    global PREDICT_FEATURES, DISTRICTS, CROPS_PREDICT, SEASONS
    if PREDICT_FEATURES is None:
        model = load_crop_predict_model()
        PREDICT_FEATURES = list(model.feature_names_in_)
        
        def extract_categories(prefix):
            return sorted(set(
                f[len(prefix):] for f in PREDICT_FEATURES if f.startswith(prefix)
            ))
        
        DISTRICTS = extract_categories("District_")
        CROPS_PREDICT = extract_categories("Crop_")
        SEASONS = extract_categories("Season_")
    
    return PREDICT_FEATURES, DISTRICTS, CROPS_PREDICT, SEASONS



@app.route("/")
def index():
    _, districts, crops_predict, seasons = get_predict_features()
    return render_template("index.html",
                           districts=districts,
                           crops=crops_predict,
                           seasons=seasons,
                           crop_labels=CROP_LABELS)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Recommend suitable crops based on soil & climate parameters."""
    data = request.json
    try:
        features = pd.DataFrame([[
            float(data["nitrogen"]),
            float(data["phosphorus"]),
            float(data["potassium"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]], columns=["Nitrogen", "phosphorus", "potassium",
                     "temperature", "humidity", "ph", "rainfall"])

        model = load_crop_recom_model()
        probas = model.predict_proba(features)
        scores = {CROP_LABELS[i]: float(p[0][1]) for i, p in enumerate(probas)}
        top_crops = sorted(scores.items(), key=lambda x: -x[1])

        results = []
        for crop, score in top_crops:
            if score > 0:
                results.append({
                    "crop": crop,
                    "score": round(score * 100, 1),
                    "emoji": CROP_EMOJIS.get(crop, "🌿")
                })

        if not results:
           
            results = [{"crop": c, "score": round(s * 100, 1),
                        "emoji": CROP_EMOJIS.get(c, "🌿")}
                       for c, s in top_crops[:5]]

        return jsonify({"success": True, "recommendations": results[:10]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict crop production given district, crop, season, year and area."""
    data = request.json
    try:
        district = data["district"]
        crop = data["crop"]
        season = data["season"]
        year = float(data["year"])
        area = float(data["area"])

        # Get features and model
        predict_features, _, _, _ = get_predict_features()
        model = load_crop_predict_model()

        # Build feature vector matching crop_predict model
        row = {f: 0 for f in predict_features}
        row["Year"] = year
        row["Area"] = area

        dist_key = f"District_{district}"
        crop_key = f"Crop_{crop}"
        season_key = f"Season_{season}"

        if dist_key in row:
            row[dist_key] = 1
        if crop_key in row:
            row[crop_key] = 1
        if season_key in row:
            row[season_key] = 1

        df = pd.DataFrame([row])[predict_features]
        prediction = model.predict(df)[0]

        return jsonify({
            "success": True,
            "production": round(float(prediction), 2),
            "unit": "metric tonnes",
            "district": district,
            "crop": crop,
            "season": season,
            "year": int(year),
            "area": area
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/meta")
def meta():
    _, districts, crops_predict, seasons = get_predict_features()
    return jsonify({
        "districts": districts,
        "crops": crops_predict,
        "seasons": seasons,
        "crop_labels": CROP_LABELS
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
