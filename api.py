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

# kNN model artifacts (Notebook 3)
knn_classifier_optimum = joblib.load('./model/knn_classifier_optimum.pkl')
knn_scaler = joblib.load('./model/scaler_3.pkl')
knn_onehot_encoder = joblib.load('./model/onehot_encoder_3.pkl')

# k-Means clustering artifacts (Notebook 6)
# The KMeans model predicts which customer segment (cluster 0–4) a client belongs to.
# Features used: Age, Annual Income (k$), Spending Score (1-100)
# Cluster profiles (from elbow method optimal k=5):
#   Cluster 0 – Young, high income,  targeted premium spenders
#   Cluster 1 – Young, low-medium income, average spenders
#   Cluster 2 – Mature, high income, low spenders
#   Cluster 3 – Mature, female-majority, low income, frugal
#   Cluster 4 – Middle-aged, medium income, luxury shoppers
kmeans_model = joblib.load('./model/kmeans_model_6.pkl')
kmeans_scaler = joblib.load('./model/scaler_6.pkl')

# Association Rule Learning artifacts (Notebook 7a)
# Rules were mined from the Hahsler et al. (2011) "groceries" dataset
# (9,835 market basket transactions) using the Apriori algorithm.
# The top rules (by lift then confidence) are stored in a CSV file,
# with antecedents and consequents as frozenset strings.
import ast
import re

def _parse_frozenset(s: str) -> frozenset:
    """Parse a frozenset string like "frozenset({'item1', 'item2'})" back to frozenset."""
    match = re.match(r"frozenset\((.*)\)", str(s).strip())
    if match:
        return frozenset(ast.literal_eval(match.group(1)))
    # Fallback: plain set string e.g. "{'item1'}"
    return frozenset(ast.literal_eval(str(s).strip()))

_raw_rules = pd.read_csv('./rule/top_rules_7a.csv')
_raw_rules['antecedents'] = _raw_rules['antecedents'].apply(_parse_frozenset)
_raw_rules['consequents'] = _raw_rules['consequents'].apply(_parse_frozenset)
association_rules_df = _raw_rules  # columns: antecedents, consequents, support, confidence, lift

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
#  -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

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

@app.route('/api/v1/models/knn-classifier/predictions', methods=['POST'])
def predict_knn_classifier():
    """
    Predict late delivery risk using the optimum kNN classifier.

    Expected JSON input:
    {
        "Days for shipping (real)": 3,
        "Days for shipment (scheduled)": 4,
        "Order Item Quantity": 1,
        "Sales": 250.00,
        "Order Profit Per Order": 64.17,
        "Shipping Mode": "Second Class"
    }

    Valid values for "Shipping Mode": "First Class", "Same Day",
    "Second Class", "Standard Class"

    Returns:
    {
        "Predicted_Late_Delivery_Risk": 1,
        "Probability_On_Time (Class 0)": 0.33,
        "Probability_Late_Delivery (Class 1)": 0.67
    }
    """
    data = request.get_json()

    # Build a single-row DataFrame with the raw features
    new_data = pd.DataFrame([{
        'Days for shipping (real)':      data.get('Days for shipping (real)'),
        'Days for shipment (scheduled)': data.get('Days for shipment (scheduled)'),
        'Order Item Quantity':           data.get('Order Item Quantity'),
        'Sales':                         data.get('Sales'),
        'Order Profit Per Order':        data.get('Order Profit Per Order'),
        'Shipping Mode':                 data.get('Shipping Mode')
    }])

    # One-hot encode 'Shipping Mode' using the saved encoder (drop='first')
    encoded = knn_onehot_encoder.transform(new_data[['Shipping Mode']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=knn_onehot_encoder.get_feature_names_out(['Shipping Mode']),
        index=new_data.index
    )

    # Drop the original categorical column and append the encoded columns
    new_data = pd.concat(
        [new_data.drop('Shipping Mode', axis=1), encoded_df],
        axis=1
    )

    # Scale features using the saved StandardScaler
    new_data_scaled = knn_scaler.transform(new_data)

    # Predict class and class probabilities
    prediction   = knn_classifier_optimum.predict(new_data_scaled)[0]
    probabilities = knn_classifier_optimum.predict_proba(new_data_scaled)[0]

    return jsonify({
        'Predicted_Late_Delivery_Risk':      int(prediction),
        'Probability_On_Time (Class 0)':     round(float(probabilities[0]), 4),
        'Probability_Late_Delivery (Class 1)': round(float(probabilities[1]), 4)
    })

# *1* Sample JSON POST values
# {
#     "Days for shipping (real)": 3,
#     "Days for shipment (scheduled)": 4,
#     "Order Item Quantity": 1,
#     "Sales": 250.00,
#     "Order Profit Per Order": 64.17,
#     "Shipping Mode": "Second Class"
# }

# *2.a.* Sample cURL POST values (without HTTPS)

# curl -X POST http://127.0.0.1:5000/api/v1/models/knn-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Days for shipping (real)\": 3, \"Days for shipment (scheduled)\": 4, \
#        \"Order Item Quantity\": 1, \"Sales\": 250.00, \
#        \"Order Profit Per Order\": 64.17, \"Shipping Mode\": \"Second Class\"}"

# *2.b.* Sample cURL POST values (with HTTPS in NGINX and Gunicorn)

# curl --insecure -X POST https://127.0.0.1/api/v1/models/knn-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Days for shipping (real)\": 3, \"Days for shipment (scheduled)\": 4, \
#        \"Order Item Quantity\": 1, \"Sales\": 250.00, \
#        \"Order Profit Per Order\": 64.17, \"Shipping Mode\": \"Second Class\"}"

# *3* Sample PowerShell values:

# $body = @{
#     "Days for shipping (real)"      = 3
#     "Days for shipment (scheduled)" = 4
#     "Order Item Quantity"           = 1
#     "Sales"                         = 250.00
#     "Order Profit Per Order"        = 64.17
#     "Shipping Mode"                 = "Second Class"
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/knn-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"


# ---------------------------------------------------------------------------
# Endpoint: k-Means Cluster Classifier (Notebook 6)
# ---------------------------------------------------------------------------

@app.route('/api/v1/models/kmeans-classifier/predictions', methods=['POST'])
def predict_kmeans_cluster():
    """
    Predict the customer segment (cluster) a client belongs to using the
    k-Means model trained on the Mall Customers dataset (k=5, random_state=53).

    The model was fitted on three standardised features:
        - Age
        - Annual Income (k$)
        - Spending Score (1-100)

    Expected JSON input:
    {
        "Age": 30,
        "Annual Income (k$)": 60,
        "Spending Score (1-100)": 45
    }

    Returns:
    {
        "Predicted_Cluster": 1,
        "Cluster_Profile": "Young, low-medium income, average spenders"
    }

    Cluster profiles (derived from post-hoc analysis of cluster centroids):
        0 – Young, high income, targeted premium spenders
        1 – Young, low-medium income, average spenders
        2 – Mature, high income, low spenders
        3 – Mature, female-majority, low income, frugal
        4 – Middle-aged, medium income, luxury shoppers
    """
    # Human-readable profiles for each cluster label
    CLUSTER_PROFILES = {
        0: "Young, high income, targeted premium spenders",
        1: "Young, low-medium income, average spenders",
        2: "Mature, high income, low spenders",
        3: "Mature, female-majority, low income, frugal",
        4: "Middle-aged, medium income, luxury shoppers"
    }

    data = request.get_json()

    # Build a single-row DataFrame with the features used during training
    new_data = pd.DataFrame([{
        'Age':                        data.get('Age'),
        'Annual Income (k$)':         data.get('Annual Income (k$)'),
        'Spending Score (1-100)':     data.get('Spending Score (1-100)')
    }])

    # Apply the same StandardScaler that was fitted during training
    new_data_scaled = kmeans_scaler.transform(new_data)

    # Predict the cluster label
    cluster = int(kmeans_model.predict(new_data_scaled)[0])

    return jsonify({
        'Predicted_Cluster':  cluster,
        'Cluster_Profile':    CLUSTER_PROFILES.get(cluster, 'Unknown')
    })

# *1* Sample JSON POST values
# {
#     "Age": 30,
#     "Annual Income (k$)": 60,
#     "Spending Score (1-100)": 45
# }

# *2.a.* Sample cURL POST values (without HTTPS)
#
# curl -X POST http://127.0.0.1:5000/api/v1/models/kmeans-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Age\": 30, \"Annual Income (k$)\": 60, \"Spending Score (1-100)\": 45}"

# *2.b.* Sample cURL POST values (with HTTPS via NGINX and Gunicorn)
#
# curl --insecure -X POST https://127.0.0.1/api/v1/models/kmeans-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Age\": 30, \"Annual Income (k$)\": 60, \"Spending Score (1-100)\": 45}"

# *3* Sample PowerShell values:
#
# $body = @{
#     "Age"                    = 30
#     "Annual Income (k$)"     = 60
#     "Spending Score (1-100)" = 45
# } | ConvertTo-Json
#
# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/kmeans-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"


# ---------------------------------------------------------------------------
# Endpoint: Association Rule Recommender (Notebook 7a)
# ---------------------------------------------------------------------------

@app.route('/api/v1/models/association-rules/recommendations', methods=['POST'])
def recommend_products():
    """
    Recommend grocery products based on association rules mined with the
    Apriori algorithm on the Hahsler et al. (2011) "groceries" dataset
    (9,835 market basket transactions).

    Rules are loaded from './rule/top_rules_7a.csv' and ranked by lift then
    confidence.  For every rule whose antecedent is a subset of the client's
    current cart, the consequent items are added to the recommendation list
    (items already in the cart are excluded from the output).

    Expected JSON input:
    {
        "cart": ["matoke", "maziwa mala"]
    }

    Returns:
    {
        "cart": ["matoke", "maziwa mala"],
        "recommendations": ["beans"],
        "rules_matched": [
            {
                "antecedents": ["maziwa mala"],
                "consequents": ["beans"],
                "support": 0.4,
                "confidence": 1.0,
                "lift": 1.25
            }
        ]
    }

    If no rules match the cart, an empty recommendations list is returned.
    """
    data = request.get_json()

    cart_list = data.get('cart', [])
    if not isinstance(cart_list, list):
        return jsonify({'error': "'cart' must be a JSON array of item strings."}), 400

    cart = set(item.strip().lower() for item in cart_list)

    recommendations = set()
    matched_rules = []

    for _, rule in association_rules_df.iterrows():
        # A rule fires when every antecedent item is present in the cart
        if rule['antecedents'].issubset(cart):
            new_items = rule['consequents'] - cart  # exclude items already in cart
            if new_items:
                recommendations.update(new_items)
                matched_rules.append({
                    'antecedents': sorted(rule['antecedents']),
                    'consequents': sorted(rule['consequents']),
                    'support':     round(float(rule['support']),    4),
                    'confidence':  round(float(rule['confidence']), 4),
                    'lift':        round(float(rule['lift']),        4)
                })

    return jsonify({
        'cart':            sorted(cart),
        'recommendations': sorted(recommendations),
        'rules_matched':   matched_rules
    })

# *1* Sample JSON POST values
# {
#     "cart": ["matoke", "maziwa mala"]
# }

# *2.a.* Sample cURL POST values (without HTTPS)
#
# curl -X POST http://127.0.0.1:5000/api/v1/models/association-rules/recommendations \
#   -H "Content-Type: application/json" \
#   -d "{\"cart\": [\"matoke\", \"maziwa mala\"]}"

# *2.b.* Sample cURL POST values (with HTTPS via NGINX and Gunicorn)
#
# curl --insecure -X POST https://127.0.0.1/api/v1/models/association-rules/recommendations \
#   -H "Content-Type: application/json" \
#   -d "{\"cart\": [\"matoke\", \"maziwa mala\"]}"

# *3* Sample PowerShell values:
#
# $body = @{ "cart" = @("matoke", "maziwa mala") } | ConvertTo-Json
#
# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/association-rules/recommendations `
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
if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)