from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
nb_model = joblib.load("model/naive_Bayes_classifier_optimum.pkl")
knn_model = joblib.load("model/knn_classifier_optimum.pkl")
svm_model = joblib.load("model/support_vector_classifier_optimum.pkl")
rf_model = joblib.load("model/random_forest_classifier_optimum.pkl")
rf_scaler = joblib.load("model/scaler_5.pkl")
rf_label_encoders = joblib.load("model/label_encoders_5.pkl")

try:
    association_rules = pd.read_pickle("model/association_rules.pkl")
except FileNotFoundError:
    association_rules = None

RF_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

RF_CATEGORICAL = ["VisitorType", "Weekend", "Month"]

# ----------------------------
# Utility Function
# ----------------------------
def validate_input(data):
    if not data:
        return False, "Missing JSON body"

    if len(data) != 4:
        return False, "Expected 4 features"

    return True, None


# ----------------------------
# Naive Bayes
# ----------------------------
@app.route("/predict/naive_bayes", methods=["POST"])
def predict_nb():
    data = request.json
    valid, error = validate_input(data)

    if not valid:
        return jsonify({"error": error}), 400

    prediction = nb_model.predict([list(data.values())])
    return jsonify({"prediction": int(prediction[0])})


# ----------------------------
# kNN
# ----------------------------
@app.route("/predict/knn", methods=["POST"])
def predict_knn():
    data = request.json
    valid, error = validate_input(data)

    if not valid:
        return jsonify({"error": error}), 400

    prediction = knn_model.predict([list(data.values())])
    return jsonify({"prediction": int(prediction[0])})


# ----------------------------
# SVM
# ----------------------------
@app.route("/predict/svm", methods=["POST"])
def predict_svm():
    data = request.json
    valid, error = validate_input(data)

    if not valid:
        return jsonify({"error": error}), 400

    prediction = svm_model.predict([list(data.values())])
    return jsonify({"prediction": int(prediction[0])})


# ----------------------------
# Random Forest
# ----------------------------
@app.route("/predict/random_forest", methods=["POST"])
def predict_rf():
    data = request.json

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    missing = [f for f in RF_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        df = pd.DataFrame([data])

        # Encode categorical columns
        for col in RF_CATEGORICAL:
            df[col] = rf_label_encoders[col].transform(df[col])

        # Ensure correct order
        df = df[RF_FEATURES]

        # Scale
        df_scaled = rf_scaler.transform(df)

        prediction = rf_model.predict(df_scaled)
        probability = rf_model.predict_proba(df_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": probability[0].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ----------------------------
# Recommender
# ----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    if association_rules is None:
        return jsonify({"error": "Association rules file not found"}), 500

    data = request.json

    if not data or "items" not in data:
        return jsonify({"error": "Missing items list"}), 400

    user_items = set(data["items"])
    recommendations = []

    for _, row in association_rules.iterrows():
        antecedent = set(row["antecedents"])
        consequent = set(row["consequents"])

        if antecedent.issubset(user_items):
            recommendations.extend(list(consequent))

    return jsonify({"recommendations": list(set(recommendations))})



@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API is running",
        "endpoints": [
            "/predict/naive_bayes",
            "/predict/knn",
            "/predict/svm",
            "/predict/random_forest",
            "/recommend"
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
    
    
    