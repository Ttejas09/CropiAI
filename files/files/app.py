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

# Model URLs from Hugging Face
HF_BASE = "https://huggingface.co/TheTejas09/cropAi/resolve/main"
MODEL_URLS = {
    "Crop_Recom.pkl": f"{HF_BASE}/Crop_Recom.pkl",
    "dist_crop_season.pkl": f"{HF_BASE}/dist_crop_season.pkl",
    "crop_predict.pkl": f"{HF_BASE}/crop_predict.pkl"
}

# Cache models in memory
MODELS = {}

def download_model_safe(filename, url, timeout=60):
    """Download with timeout"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename, timeout=timeout)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def get_model(model_name):
    """Lazy load model on demand"""
    if model_name not in MODELS:
        filepath = model_name
        url = MODEL_URLS.get(model_name)
        
        if not os.path.exists(filepath):
            if not url or not download_model_safe(filepath, url):
                raise Exception(f"Cannot load {model_name}")
        
        with open(filepath, 'rb') as f:
            MODELS[model_name] = pickle.load(f)
    
    return MODELS[model_name]

# Constants - no model loading needed
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

# Cached feature info
_cached_features = None

def get_features():
    """Get model features (cached)"""
    global _cached_features
    if _cached_features is None:
        model = get_model("crop_predict.pkl")
        features = list(model.feature_names_in_)
        
        # Extract categories
        districts = sorted(set(f[9:] for f in features if f.startswith("District_")))
        crops = sorted(set(f[5:] for f in features if f.startswith("Crop_")))
        seasons = sorted(set(f[7:] for f in features if f.startswith("Season_")))
        
        _cached_features = {
            'features': features,
            'districts': districts,
            'crops': crops,
            'seasons': seasons
        }
    return _cached_features

# Routes
@app.route("/status")
def status():
    """Health check - no model loading needed"""
    return jsonify({"status": "ok", "app": "CropiAI v1.0"})

@app.route("/")
def index():
    """Main page"""
    try:
        info = get_features()
        return render_template("index.html",
                               districts=info['districts'],
                               crops=info['crops'],
                               seasons=info['seasons'],
                               crop_labels=CROP_LABELS)
    except Exception as e:
        return f"<h1>Loading models...</h1><p>{str(e)}</p>", 202

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Recommend crops based on soil/climate"""
    try:
        data = request.json
        model = get_model("Crop_Recom.pkl")
        
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
        
        probas = model.predict_proba(features)
        scores = {CROP_LABELS[i]: float(p[0][1]) for i, p in enumerate(probas)}
        top_crops = sorted(scores.items(), key=lambda x: -x[1])
        
        results = [{
            "crop": crop,
            "score": round(score * 100, 1),
            "emoji": CROP_EMOJIS.get(crop, "🌿")
        } for crop, score in top_crops if score > 0][:10]
        
        if not results:
            results = [{
                "crop": c,
                "score": round(s * 100, 1),
                "emoji": CROP_EMOJIS.get(c, "🌿")
            } for c, s in top_crops[:5]]
        
        return jsonify({"success": True, "recommendations": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict production"""
    try:
        data = request.json
        info = get_features()
        model = get_model("crop_predict.pkl")
        
        features = info['features']
        row = {f: 0 for f in features}
        row["Year"] = float(data["year"])
        row["Area"] = float(data["area"])
        row[f"District_{data['district']}"] = 1
        row[f"Crop_{data['crop']}"] = 1
        row[f"Season_{data['season']}"] = 1
        
        df = pd.DataFrame([row])[features]
        prediction = model.predict(df)[0]
        
        return jsonify({
            "success": True,
            "production": round(float(prediction), 2),
            "unit": "metric tonnes",
            "district": data["district"],
            "crop": data["crop"],
            "season": data["season"],
            "year": int(float(data["year"])),
            "area": float(data["area"])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/meta")
def meta():
    """Get metadata"""
    try:
        info = get_features()
        return jsonify({
            "districts": info['districts'],
            "crops": info['crops'],
            "seasons": info['seasons'],
            "crop_labels": CROP_LABELS
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
