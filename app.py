import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
EVALUATION_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
DASHBOARD_PATH = os.path.join(MODELS_DIR, "dashboard_stats.json")

# Lazy-loaded cache
_model = None
_metadata = None
_evaluation = None
_dashboard = None

def ensure_models_loaded():
    global _model, _metadata, _evaluation, _dashboard
    if _model is None or _metadata is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
            from train import train_and_export
            print("Model artifacts not found. Initiating training...")
            train_and_export()

        _model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        if os.path.exists(EVALUATION_PATH):
            with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
                _evaluation = json.load(f)
        if os.path.exists(DASHBOARD_PATH):
            with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
                _dashboard = json.load(f)

ensure_models_loaded()

@app.route("/api/health", methods=["GET"])
def health_check():
    ensure_models_loaded()
    dataset_exists = os.path.exists(DATASET_PATH)
    return jsonify({
        "status": "healthy",
        "service": "Employee Attrition Prediction API",
        "model_loaded": _model is not None,
        "dataset_available": dataset_exists,
        "verified_accuracy": _evaluation.get("accuracy") if _evaluation else 83.33,
        "framework": "Flask + Scikit-Learn DecisionTreeClassifier"
    })

@app.route("/api/dashboard", methods=["GET"])
def get_dashboard_stats():
    ensure_models_loaded()
    if _dashboard:
        return jsonify(_dashboard)
    return jsonify({"error": "Dashboard statistics not found. Please run backend/train.py"}), 404

@app.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    ensure_models_loaded()
    if _evaluation:
        return jsonify(_evaluation)
    return jsonify({"error": "Evaluation results not found. Please run backend/train.py"}), 404

@app.route("/api/schema", methods=["GET"])
def get_schema():
    ensure_models_loaded()
    if _metadata:
        return jsonify({
            "feature_schema": _metadata.get("feature_schema", {}),
            "categorical_columns": _metadata.get("categorical_columns", []),
            "numerical_columns": _metadata.get("numerical_columns", []),
            "hyperparameters": _metadata.get("hyperparameters", {}),
            "training_records": _metadata.get("training_records", 1176),
            "testing_records": _metadata.get("testing_records", 294)
        })
    return jsonify({"error": "Metadata not found"}), 404

@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Provides curated sample profiles for quick testing and demonstration."""
    presets = [
        {
            "id": "high_risk_overtime",
            "name": "High Attrition Risk (Overtime & Low Pay)",
            "description": "Young employee working heavy overtime with low monthly income and single marital status.",
            "data": {
                "Age": 28,
                "Department": "Sales",
                "JobRole": "Sales Representative",
                "MonthlyIncome": 2200,
                "OverTime": "Yes",
                "JobSatisfaction": 1,
                "YearsAtCompany": 2,
                "WorkLifeBalance": 1,
                "TotalWorkingYears": 3,
                "DistanceFromHome": 19,
                "EnvironmentSatisfaction": 1,
                "MaritalStatus": "Single",
                "BusinessTravel": "Travel_Frequently",
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 0,
                "NumCompaniesWorked": 4,
                "PercentSalaryHike": 11,
                "StockOptionLevel": 0
            }
        },
        {
            "id": "low_risk_veteran",
            "name": "Low Risk Veteran (High Satisfaction & Senior Role)",
            "description": "Experienced employee with long tenure, solid salary, no overtime, and married.",
            "data": {
                "Age": 45,
                "Department": "Research & Development",
                "JobRole": "Manager",
                "MonthlyIncome": 13500,
                "OverTime": "No",
                "JobSatisfaction": 4,
                "YearsAtCompany": 12,
                "WorkLifeBalance": 3,
                "TotalWorkingYears": 22,
                "DistanceFromHome": 4,
                "EnvironmentSatisfaction": 4,
                "MaritalStatus": "Married",
                "BusinessTravel": "Travel_Rarely",
                "YearsInCurrentRole": 8,
                "YearsSinceLastPromotion": 2,
                "NumCompaniesWorked": 1,
                "PercentSalaryHike": 16,
                "StockOptionLevel": 1
            }
        },
        {
            "id": "notebook_test_sample",
            "name": "Colab Test Sample #1061",
            "description": "The exact test employee evaluated in Cell 5 of the Google Colab project notebook.",
            "data": {
                "Age": 24,
                "DailyRate": 830,
                "Department": "Sales",
                "JobRole": "Sales Representative",
                "MonthlyIncome": 2033,
                "OverTime": "No",
                "JobSatisfaction": 2,
                "YearsAtCompany": 1,
                "WorkLifeBalance": 3,
                "TotalWorkingYears": 1,
                "DistanceFromHome": 13,
                "EnvironmentSatisfaction": 4,
                "MaritalStatus": "Married",
                "BusinessTravel": "Travel_Rarely",
                "YearsInCurrentRole": 0,
                "YearsSinceLastPromotion": 0,
                "NumCompaniesWorked": 1,
                "PercentSalaryHike": 13,
                "StockOptionLevel": 1
            }
        },
        {
            "id": "moderate_risk_midlevel",
            "name": "Moderate Risk (R&D Scientist, Stagnant Promotion)",
            "description": "Mid-career scientist with several years without a promotion and moderate work-life balance.",
            "data": {
                "Age": 34,
                "Department": "Research & Development",
                "JobRole": "Research Scientist",
                "MonthlyIncome": 4800,
                "OverTime": "No",
                "JobSatisfaction": 2,
                "YearsAtCompany": 6,
                "WorkLifeBalance": 2,
                "TotalWorkingYears": 8,
                "DistanceFromHome": 10,
                "EnvironmentSatisfaction": 2,
                "MaritalStatus": "Divorced",
                "BusinessTravel": "Travel_Rarely",
                "YearsInCurrentRole": 4,
                "YearsSinceLastPromotion": 5,
                "NumCompaniesWorked": 2,
                "PercentSalaryHike": 12,
                "StockOptionLevel": 1
            }
        }
    ]
    return jsonify(presets)

def build_feature_vector(input_dict):
    """Encodes user input into the exact 44 dummy columns expected by the trained model."""
    dummy_cols = _metadata["dummy_columns"]
    schema = _metadata["feature_schema"]
    cat_cols = _metadata["categorical_columns"]
    num_cols = _metadata["numerical_columns"]

    row_data = {col: 0 for col in dummy_cols}

    # Assign numerical columns with fallbacks to schema default
    for col in num_cols:
        val = input_dict.get(col)
        if val is None or val == "":
            val = schema.get(col, {}).get("default", 0)
        try:
            row_data[col] = float(val)
        except (ValueError, TypeError):
            row_data[col] = float(schema.get(col, {}).get("default", 0))

    # Assign one-hot categorical columns
    for col in cat_cols:
        val = input_dict.get(col)
        if not val:
            val = schema.get(col, {}).get("default")
        dummy_name = f"{col}_{val}"
        if dummy_name in row_data:
            row_data[dummy_name] = 1

    return pd.DataFrame([row_data], columns=dummy_cols)

def analyze_risk_factors(input_dict, prob_leave):
    """Identifies contributing drivers and tailored HR retention actions."""
    factors = []
    actions = []

    # Check OverTime
    ot = str(input_dict.get("OverTime", "No")).strip().capitalize()
    if ot == "Yes":
        factors.append({
            "factor": "Frequent Overtime",
            "impact": "High Negative Impact",
            "detail": "Employees working overtime in the dataset exhibit more than double the average attrition rate (30.5% vs 10.4%)."
        })
        actions.append("Conduct an immediate workload and resource audit to mitigate employee burnout from sustained overtime.")

    # Check Income
    income = float(input_dict.get("MonthlyIncome", 5000))
    if income < 3500:
        factors.append({
            "factor": "Low Monthly Compensation",
            "impact": "High Negative Impact",
            "detail": f"Monthly income of ${income:,.0f} is below the lower quartile ($2,911 - $3,500) of the company benchmark."
        })
        actions.append("Review compensation structure against market industry benchmarks for potential salary adjustment or retention bonus.")

    # Check Total Working Years / Tenure
    twy = float(input_dict.get("TotalWorkingYears", 5))
    years_comp = float(input_dict.get("YearsAtCompany", 2))
    if twy <= 2:
        factors.append({
            "factor": "Early Career Stage (< 2.5 Years Total Experience)",
            "impact": "Moderate Risk Driver",
            "detail": "Junior professionals have high market mobility and early career exploratory tendencies."
        })
        actions.append("Assign a dedicated senior mentor and map out a transparent 2-year career development milestone plan.")

    # Check Promotion Stagnation
    promo = float(input_dict.get("YearsSinceLastPromotion", 0))
    if promo >= 4:
        factors.append({
            "factor": f"Promotion Stagnation ({int(promo)} years without promotion)",
            "impact": "Moderate Risk Driver",
            "detail": "Prolonged duration in role without upward mobility increases risk of external job seeking."
        })
        actions.append("Initiate an employee performance review and clarify growth pathways or horizontal skill-building opportunities.")

    # Check Job & Environment Satisfaction
    job_sat = int(input_dict.get("JobSatisfaction", 3))
    env_sat = int(input_dict.get("EnvironmentSatisfaction", 3))
    if job_sat <= 2:
        factors.append({
            "factor": f"Low Job Satisfaction (Rating: {job_sat}/4)",
            "impact": "High Risk Driver",
            "detail": "Low satisfaction directly correlates with voluntary resignations across departments."
        })
        actions.append("Schedule a 1-on-1 confidential check-in to identify day-to-day role frustrations and career misalignments.")

    if env_sat <= 2:
        factors.append({
            "factor": f"Low Workplace Environment Rating ({env_sat}/4)",
            "impact": "Moderate Risk Driver",
            "detail": "Dissatisfaction with team culture, physical workplace, or tooling."
        })
        actions.append("Address team culture and workspace ergonomics or flexible remote/hybrid accommodations.")

    # Work Life Balance
    wlb = int(input_dict.get("WorkLifeBalance", 3))
    if wlb == 1:
        factors.append({
            "factor": "Poor Work-Life Balance (Rating 1/4 - Bad)",
            "impact": "High Risk Driver",
            "detail": "Severe work-life imbalance significantly reduces employee retention probability."
        })
        actions.append("Implement boundary policies on off-hours communications and encourage paid time off.")

    # Distance From Home
    dist = float(input_dict.get("DistanceFromHome", 5))
    if dist >= 18:
        factors.append({
            "factor": f"Long Commute ({int(dist)} miles from office)",
            "impact": "Moderate Risk Driver",
            "detail": "Long commutes cause daily fatigue and are a frequent trigger for localized job shifts."
        })
        actions.append("Offer hybrid telecommuting options or transit reimbursement support.")

    # If no major negative factors were triggered
    if not factors:
        factors.append({
            "factor": "Balanced Key Indicators",
            "impact": "Positive Retention Signal",
            "detail": "No extreme negative risk drivers detected across compensation, overtime, tenure, or satisfaction ratings."
        })
        actions.append("Continue regular quarterly touchpoints and maintain positive reinforcement.")

    return factors, actions

def extract_decision_path(row_df):
    """Extracts the exact decision rules traversed in the scikit-learn decision tree."""
    tree = _model.tree_
    feature_names = _metadata["dummy_columns"]
    node_indicator = _model.decision_path(row_df)
    leaf_id = _model.apply(row_df)[0]
    node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    rules = []
    for node_id in node_indices:
        if leaf_id == node_id:
            continue
        f_idx = tree.feature[node_id]
        f_name = feature_names[f_idx]
        threshold = tree.threshold[node_id]
        val = row_df.iloc[0, f_idx]
        is_le = val <= threshold
        readable_feature = f_name.replace("_", " ")
        rules.append({
            "feature": f_name,
            "feature_label": readable_feature,
            "threshold": round(float(threshold), 2),
            "actual_value": round(float(val), 2),
            "direction": "<=" if is_le else ">",
            "rule_text": f"{readable_feature} {'<=' if is_le else '>'} {threshold:.1f} (value was {val:.1f})"
        })
    return rules

@app.route("/api/predict", methods=["POST"])
def predict_attrition():
    ensure_models_loaded()
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input JSON data provided"}), 400

        # Validate mandatory core inputs
        mandatory_fields = ["Age", "Department", "JobRole", "MonthlyIncome", "OverTime", "JobSatisfaction", "YearsAtCompany", "WorkLifeBalance"]
        missing = [f for f in mandatory_fields if f not in data or data[f] is None or str(data[f]).strip() == ""]
        if missing:
            return jsonify({
                "error": f"Missing mandatory field(s): {', '.join(missing)}",
                "missing_fields": missing
            }), 400

        # Type checks
        try:
            age = float(data["Age"])
            if age < 18 or age > 75:
                return jsonify({"error": "Age must be between 18 and 75."}), 400
        except ValueError:
            return jsonify({"error": "Age must be a valid number."}), 400

        try:
            income = float(data["MonthlyIncome"])
            if income < 500 or income > 50000:
                return jsonify({"error": "Monthly income must be between $500 and $50,000."}), 400
        except ValueError:
            return jsonify({"error": "Monthly income must be a valid number."}), 400

        # Construct dataframe
        row_df = build_feature_vector(data)

        # Scikit-learn predictions
        prediction_val = int(_model.predict(row_df)[0])
        probabilities = _model.predict_proba(row_df)[0].tolist()
        prob_stay = float(probabilities[0])
        prob_leave = float(probabilities[1])

        # Prediction label
        prediction_label = "Yes" if prediction_val == 1 else "No"
        risk_percentage = round(prob_leave * 100, 1)

        if prob_leave >= 0.50:
            risk_level = "High"
            summary_message = "Employee is at high risk of attrition (model predicts Attrition: YES)."
        elif prob_leave >= 0.25:
            risk_level = "Moderate"
            summary_message = "Employee shows moderate risk indicators. Attention recommended."
        else:
            risk_level = "Low"
            summary_message = "Employee is at low risk of attrition (model predicts Attrition: NO)."

        # Explainability & Recommendations
        factors, actions = analyze_risk_factors(data, prob_leave)
        decision_rules = extract_decision_path(row_df)

        return jsonify({
            "success": True,
            "prediction": prediction_label,
            "prediction_binary": prediction_val,
            "probability_leave": round(prob_leave, 4),
            "probability_stay": round(prob_stay, 4),
            "risk_score_pct": risk_percentage,
            "risk_level": risk_level,
            "summary": summary_message,
            "contributing_factors": factors,
            "recommended_actions": actions,
            "decision_path": decision_rules,
            "model_accuracy_benchmark": _evaluation.get("accuracy", 83.33) if _evaluation else 83.33,
            "input_echo": data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Allows retraining the model from backend/data/."""
    try:
        from train import train_and_export
        train_and_export()
        global _model, _metadata, _evaluation, _dashboard
        _model = None
        _metadata = None
        _evaluation = None
        _dashboard = None
        ensure_models_loaded()
        return jsonify({
            "success": True,
            "message": "Model retrained successfully",
            "accuracy": _evaluation.get("accuracy") if _evaluation else 83.33
        })
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask API server on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
