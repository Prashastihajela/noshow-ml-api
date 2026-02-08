import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# load trained model
with open("xgb_noshow_model.pkl", "rb") as f:
    model = pickle.load(f)

def build_features(data):
    # initialize all expected features to 0
    features = {col: 0 for col in model.feature_names_in_}

    # direct features
    features["Scholarship"] = data["Scholarship"]
    features["Hypertension"] = data["Hypertension"]
    features["Diabetes"] = data["Diabetes"]
    features["Handicap"] = data["Handicap"]
    features["Alcoholism"] = data["Alcoholism"]
    features["Age"] = data["Age"]
    features["Past_NoShow_Count"] = data["Past_NoShow_Count"]
    features["Medical_Transport"] = data["Medical_Transport"]
    features["Gender_M"] = data["Gender_M"]
    features["WaitingDays"] = data["WaitingDays"]

    # derived
    features["SMS_received"] = 1 if data["Reminder_channel"] == "SMS" else 0
    features["Visit_Number"] = data["Past_NoShow_Count"] + 1
    features["Reliability_Score"] = max(
        0,
        1 - (data["Past_NoShow_Count"] * 0.2)
    )

    # handicap bucket used during training
    features["Handicap/Old_Neither"] = 1

    # appointment day (only if model knows it)
    day_col = f"Appointment_Day_{data['Appointment_Day']}"
    if day_col in features:
        features[day_col] = 1
    # else: Friday / Sunday → all zeros (expected)

    return features


@app.route("/predict", methods=["POST"])
def predict():
    raw_data = request.json

    feature_dict = build_features(raw_data)

    df = pd.DataFrame([feature_dict])

    df = df[model.feature_names_in_]

    risk = model.predict_proba(df)[0][1]

    return jsonify({
        "noshow_risk": round(float(risk), 2)
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
