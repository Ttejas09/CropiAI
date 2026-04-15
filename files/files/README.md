# KrishiMind — Intelligent Crop Advisory

A deployable Flask web application using your three ML models:
- **Crop_Recom.pkl** → Recommends crops from 7 soil/climate inputs
- **crop_predict.pkl** → Predicts production volume (district + crop + season + year + area)
- **dist_crop_season.pkl** → ColumnTransformer for encoding (used internally by crop_predict)

---

## Project Structure

```
agri_app/
├── app.py                  # Flask backend (API + routes)
├── Crop_Recom.pkl          # Crop recommendation model
├── crop_predict.pkl        # Production prediction model
├── dist_crop_season.pkl    # Preprocessor transformer
├── requirements.txt
├── Procfile                # For Render / Railway / Heroku
└── templates/
    └── index.html          # Single-page frontend
```

---

## Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run dev server
python app.py
# → Open http://localhost:5000
```

---

## Production Deployment

### Option A — Render.com (Free tier, recommended)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python version**: 3.11
5. Deploy ✅

### Option B — Railway.app

1. Push to GitHub
2. New project → Deploy from GitHub repo
3. Railway auto-detects Flask via Procfile
4. Set environment variable `PORT=5000` if needed

### Option C — Heroku

```bash
heroku create krishimind
git push heroku main
heroku open
```

### Option D — VPS / EC2 (nginx + gunicorn)

```bash
# Install
pip install -r requirements.txt

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# nginx config (place in /etc/nginx/sites-available/krishimind)
server {
    listen 80;
    server_name yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## API Endpoints

### `POST /api/recommend`
Recommend crops based on soil/climate parameters.

**Request:**
```json
{
  "nitrogen": 90,
  "phosphorus": 42,
  "potassium": 43,
  "temperature": 25.0,
  "humidity": 80.0,
  "ph": 6.5,
  "rainfall": 150.0
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {"crop": "rice", "score": 100.0, "emoji": "🌾"},
    {"crop": "jute", "score": 85.0, "emoji": "🌾"}
  ]
}
```

---

### `POST /api/predict`
Predict crop production for a district-crop-season combination.

**Request:**
```json
{
  "district": "PUNE",
  "crop": "Rice",
  "season": "Kharif",
  "year": 2024,
  "area": 500
}
```

**Response:**
```json
{
  "success": true,
  "production": 12540.75,
  "unit": "metric tonnes",
  "district": "PUNE",
  "crop": "Rice",
  "season": "Kharif",
  "year": 2024,
  "area": 500.0
}
```

---

### `GET /api/meta`
Returns all valid districts, crops, and seasons for the predictor dropdowns.

---

## Notes

- The `dist_crop_season.pkl` is an **unfitted** ColumnTransformer used as a config reference.
  The actual one-hot encoding for `crop_predict.pkl` is done manually in `app.py`
  using the feature names embedded in the trained model itself.
- Model was trained with scikit-learn 1.7.2 — use the same version in production
  to avoid version mismatch warnings.
