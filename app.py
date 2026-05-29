from flask import Flask, jsonify

import pandas as pd
import requests
import joblib


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load(
    'fishermen_risk_model.pkl'
)


# =====================================================
# CREATE FLASK APP
# =====================================================

app = Flask(__name__)


# =====================================================
# REGIONS
# =====================================================

REGIONS = {

    "mwanza": [-2.5164, 32.9175],

    "bukoba": [-1.3317, 31.8122],

    "musoma": [-1.5000, 33.8000]
}


# =====================================================
# WEATHER API KEY
# =====================================================

API_KEY = "0f7d1611a2f3edfd5394c940457304ab"


# =====================================================
# GET WEATHER DATA
# =====================================================

def get_weather(region):

    lat = REGIONS[region][0]

    lon = REGIONS[region][1]

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {

        "lat": lat,

        "lon": lon,

        "appid": API_KEY,

        "units": "metric"
    }

    response = requests.get(
        url,
        params=params
    )

    return response.json()


# =====================================================
# PREPARE FEATURES
# =====================================================

def prepare_features(weather):

    # Weather values

    temperature = weather['main']['temp']

    humidity = weather['main']['humidity']

    pressure = weather['main']['pressure']

    wind_speed = weather['wind']['speed']

    wind_direction = weather['wind']['deg']

    visibility = weather.get(
        'visibility',
        10000
    ) / 1000


    # Rainfall

    rainfall = 0

    if 'rain' in weather:

        rainfall = weather['rain'].get(
            '1h',
            0
        )


    # Current time

    now = pd.Timestamp.now()

    hour = now.hour

    month = now.month

    dayofweek = now.dayofweek


    # Region encoding

    zone_encoded = 0


    # Engineered features

    wind_avg_3h = wind_speed

    temp_avg_3h = temperature

    rain_avg_3h = rainfall

    pressure_change = 0


    # Final dataframe

    features = pd.DataFrame({

        'temperature_C': [temperature],

        'humidity': [humidity],

        'pressure_hPa': [pressure],

        'wind_direction': [wind_direction],

        'visibility_km': [visibility],

        'hour': [hour],

        'month': [month],

        'dayofweek': [dayofweek],

        'wind_avg_3h': [wind_avg_3h],

        'temp_avg_3h': [temp_avg_3h],

        'rain_avg_3h': [rain_avg_3h],

        'pressure_change': [pressure_change],

        'zone_encoded': [zone_encoded]
    })

    return features


# =====================================================
# HOME ROUTE
# =====================================================

@app.route('/')

def home():

    return "Lake Victoria Risk Prediction API"


# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.route('/predict')

def predict():

    # Temporary fixed region for browser testing

    region = "mwanza"
    region = "bukoba"


    # Fetch weather

    weather = get_weather(region)

    print(weather)
    # Prepare features

    features = prepare_features(weather)


    # Model prediction

    prediction = model.predict(features)[0]


    # Labels
    

    labels = {

        0: "LOW",

        1: "MEDIUM",

        2: "HIGH"
    }


    risk = labels[prediction]


    # Return response

    if risk == "LOW":

        message = "Weather is safe for fishing."

    elif risk == "MEDIUM":

        message = "Moderate weather risk. Be careful."

    else:

        message = "Dangerous weather detected. Avoid fishing."


    return jsonify({

        "region": region,

        "risk_level": risk,

        "message": message
    })


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(debug=True)