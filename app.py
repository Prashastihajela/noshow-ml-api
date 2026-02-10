import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# load trained model
with open("xgb_noshow_model.pkl", "rb") as f:
    model = pickle.load(f)

def build_features(data):
    features = {}

    # direct mappings
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

    features["Visit_Number"] = data["Visit_Number"]

    features["Reliability_Score"] = max( 0, 1 - (data["Past_NoShow_Count"]/ max(data["Visit_Number"], 1)))


    # handicap age 
    is_elderly = 1 if data["Age"] >= 65 else 0
    is_disabled = 1 if data["Handicap"] == 1 else 0

    features["Handicap/Old_Neither"] = 1 if (is_elderly == 0 and is_disabled == 0) else 0
    features["Handicap/Old_Only Disabled"] = 1 if (is_elderly == 0 and is_disabled == 1) else 0
    features["Handicap/Old_Only Elderly"] = 1 if (is_elderly == 1 and is_disabled == 0) else 0


    # appointment day one-hot
    days = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Saturday"
    ]

    for day in days:
        col = f"Appointment_Day_{day}"
        features[col] = 1 if data["Appointment_Day"] == day else 0

    return features

@app.route("/predict", methods=["POST"])
def predict():
    raw_data = request.json

    feature_dict = build_features(raw_data)

    df = pd.DataFrame([feature_dict])

    df = df[model.feature_names_in_]

    risk = model.predict_proba(df, validate_features=False)[0][1]

    return jsonify({
        "noshow_risk": round(float(risk), 2)
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
