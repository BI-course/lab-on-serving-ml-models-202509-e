from flask import Flask, request, jsonify
# Cross-Origin Resource Sharing (CORS)
# Modern browsers apply the "same-origin policy", which blocks web pages from
# making requests to a different origin than the one that served the page.
# This helps prevent malicious sites from reading sensitive data from another
# site you are logged into.
#
# However, there are many legitimate cases where cross-origin requests are
# needed. One example is:
#
## Single-Page Applications (SPA) hosted at example-frontend.com need to call
## APIs hosted at api.example-backend.com.
#
# To support this safely, CORS lets servers explicitly allow such requests.
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# CORS(
#     app,
#     resources={r"/api/*": {
#         "origins": [
#             "https://127.0.0.1",
#             "https://localhost"
#         ]
#     }},
#     methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type"]
# )

CORS(
    app, supports_credentials=False,
    resources={r"/api/*": { # This means CORS will only apply to routes that start with /api/
               "origins": [
                   "https://127.0.0.1", "https://localhost",
                   "https://127.0.0.1:443", "https://localhost:443",
                   "http://127.0.0.1", "http://localhost",
                   "http://127.0.0.1:5000", "http://localhost:5000",
                   "http://127.0.0.1:5500", "http://localhost:5500"
                ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"])

# CORS(app, supports_credentials=False,
#      origins=["*"])

# Load different models
# joblib is used to load a trained model so that the API can serve ML predictions
decisiontree_classifier_baseline = joblib.load('./model/decisiontree_classifier_baseline.pkl')
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

# Naive Bayes models
naive_bayes_classifier_optimum = joblib.load('./model/naive_Bayes_classifier_optimum.pkl')
label_encoders_2 = joblib.load('./model/label_encoders_2.pkl')

# Defines an HTTP endpoint
@app.route('/api/v1/models/decision-tree-classifier/predictions', methods=['POST'])
def predict_decision_tree_classifier():
    # Accepts JSON data sent by a client (browser, curl, Postman, etc.)
    data = request.get_json()
    # Create a DataFrame with the correct feature names
    new_data = pd.DataFrame([{
        'monthly_fee': data.get('monthly_fee'),
        'customer_age': data.get('customer_age'),
        'support_calls': data.get('support_calls')
    }])

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'monthly_fee',
        'customer_age',
        'support_calls'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Performs a prediction using the already trained machine learning model
    prediction = decisiontree_classifier_baseline.predict(new_data)[0]
    
    # Returns the result as a JSON response:
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

# *2.a.* Sample cURL POST values (without HTTPS in NGINX and Gunicorn)

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *2.b.* Sample cURL POST values (with HTTPS in NGINX and Gunicorn)

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *3* Sample PowerShell values:

# $body = @{
#     monthly_fee = 60
#     customer_age = 30
#     support_calls = 1
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

@app.route('/api/v1/models/decision-tree-regressor/predictions', methods=['POST'])
def predict_decision_tree_regressor():
    data = request.get_json()
    # Expected input keys:
    # 'PaymentDate', 'CustomerType', 'BranchSubCounty',
    # 'ProductCategoryName', 'QuantityOrdered', 'PercentageProfitPerUnit'

    # Create a DataFrame based on the input
    new_data = pd.DataFrame([data])

    # Convert PaymentDate to datetime
    new_data['PaymentDate'] = pd.to_datetime(new_data['PaymentDate'])

    # Identify all datetime columns
    datetime_columns = new_data.select_dtypes(include=['datetime64']).columns

    categorical_cols = new_data.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns

    # Encode categorical columns
    for col in categorical_cols:
        if col in new_data:
            new_data[col] = label_encoders_1b[col].transform(new_data[col])

    # Feature engineering for date
    new_data['PaymentDate_year'] = new_data['PaymentDate'].dt.year # type: ignore
    new_data['PaymentDate_month'] = new_data['PaymentDate'].dt.month # type: ignore
    new_data['PaymentDate_day'] = new_data['PaymentDate'].dt.day # type: ignore
    new_data['PaymentDate_dayofweek'] = new_data['PaymentDate'].dt.dayofweek # type: ignore
    new_data = new_data.drop(columns=datetime_columns)

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'CustomerType',
        'BranchSubCounty',
        'ProductCategoryName',
        'QuantityOrdered',
        'PaymentDate_year',
        'PaymentDate_month',
        'PaymentDate_day',
        'PaymentDate_dayofweek'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Predict
    prediction = decisiontree_regressor_optimum.predict(new_data)[0]
    return jsonify({'Predicted Percentage Profit per Unit = ': float(prediction)})

# *1* Sample JSON POST values
# {
#     "CustomerType": "Business",
#     "BranchSubCounty": "Kilimani",
#     "ProductCategoryName": "Meat-Based Dishes",
#     "QuantityOrdered": 8,
#     "PaymentDate": "2027-11-13"
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *2.b.* Sample cURL POST values

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *3* Sample PowerShell values:

# $body = @{
#     PaymentDate         = "2027-11-13"
#     CustomerType        = "Business"
#     BranchSubCounty     = "Kilimani"
#     ProductCategoryName = "Meat-Based Dishes"
#     QuantityOrdered = 8
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

# This ensures the Flask web server only starts when you run this file directly
# (e.g., `python api.py`), and not if you import api.py from another script or test.

# __name__ is a special variable in Python. When you run a script directly,
# __name__ is set to '__main__'. If the script is imported, __name__ is set to
# the module's name.

# if __name__ == '__main__': checks if the script is being run directly.

# app.run(debug=True) starts the Flask development server with debugging enabled.
# This means:
## The server will automatically reload if you make code changes.
## You get detailed error messages in the browser if something goes wrong.
@app.route('/api/v1/models/naive-bayes-classifier/predictions', methods=['POST'])
def predict_naive_bayes_classifier():
    # Expected input keys (all 17 features):
    # 'Administrative', 'Administrative_Duration', 'Informational',
    # 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    # 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    # 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
    # 'VisitorType', 'Weekend'
    data = request.get_json()

    new_data = pd.DataFrame([data])

    # Encode categorical columns using the saved label encoders
    for col in ['VisitorType', 'Weekend', 'Month']:
        if col in new_data.columns:
            new_data[col] = label_encoders_2[col].transform(new_data[col])

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'Administrative', 'Administrative_Duration',
        'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'Browser', 'Region',
        'TrafficType', 'VisitorType', 'Weekend'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Predict: 0 = No Revenue, 1 = Revenue
    prediction = naive_bayes_classifier_optimum.predict(new_data)[0]
    probability = naive_bayes_classifier_optimum.predict_proba(new_data)[0].tolist()

    return jsonify({
        'Predicted Class': int(prediction),
        'Predicted Label': 'Revenue' if prediction == 1 else 'No Revenue',
        'Probability [No Revenue, Revenue]': probability
    })

# *1* Sample JSON POST values
# {
#     "Administrative": 2,
#     "Administrative_Duration": 50.0,
#     "Informational": 0,
#     "Informational_Duration": 0.0,
#     "ProductRelated": 20,
#     "ProductRelated_Duration": 400.0,
#     "BounceRates": 0.02,
#     "ExitRates": 0.05,
#     "PageValues": 0.0,
#     "SpecialDay": 0.0,
#     "Month": "Nov",
#     "OperatingSystems": 2,
#     "Browser": 1,
#     "Region": 1,
#     "TrafficType": 2,
#     "VisitorType": "Returning_Visitor",
#     "Weekend": "False"
# }

# *2.a.* Sample cURL POST values (without HTTPS)
# curl -X POST http://127.0.0.1:5000/api/v1/models/naive-bayes-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Administrative\": 2, \"Administrative_Duration\": 50.0, \"Informational\": 0, \"Informational_Duration\": 0.0, \"ProductRelated\": 20, \"ProductRelated_Duration\": 400.0, \"BounceRates\": 0.02, \"ExitRates\": 0.05, \"PageValues\": 0.0, \"SpecialDay\": 0.0, \"Month\": \"Nov\", \"OperatingSystems\": 2, \"Browser\": 1, \"Region\": 1, \"TrafficType\": 2, \"VisitorType\": \"Returning_Visitor\", \"Weekend\": \"False\"}"

# *2.b.* Sample cURL POST values (with HTTPS)
# curl --insecure -X POST https://127.0.0.1/api/v1/models/naive-bayes-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Administrative\": 2, \"Administrative_Duration\": 50.0, \"Informational\": 0, \"Informational_Duration\": 0.0, \"ProductRelated\": 20, \"ProductRelated_Duration\": 400.0, \"BounceRates\": 0.02, \"ExitRates\": 0.05, \"PageValues\": 0.0, \"SpecialDay\": 0.0, \"Month\": \"Nov\", \"OperatingSystems\": 2, \"Browser\": 1, \"Region\": 1, \"TrafficType\": 2, \"VisitorType\": \"Returning_Visitor\", \"Weekend\": \"False\"}"

# *3* Sample PowerShell values:
# $body = @{
#     Administrative           = 2
#     Administrative_Duration  = 50.0
#     Informational            = 0
#     Informational_Duration   = 0.0
#     ProductRelated           = 20
#     ProductRelated_Duration  = 400.0
#     BounceRates              = 0.02
#     ExitRates                = 0.05
#     PageValues               = 0.0
#     SpecialDay               = 0.0
#     Month                    = "Nov"
#     OperatingSystems         = 2
#     Browser                  = 1
#     Region                   = 1
#     TrafficType              = 2
#     VisitorType              = "Returning_Visitor"
#     Weekend                  = "False"
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/naive-bayes-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"


if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)
