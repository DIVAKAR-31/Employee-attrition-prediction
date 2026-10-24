# Employee Attrition Prediction Using Machine Learning
**Machine Learning Project**

A modern, full-stack, interpretable HR analytics and machine learning web application that predicts employee turnover likelihood and identifies organizational flight risks using the **IBM HR Analytics Employee Attrition dataset** and a **Scikit-Learn Decision Tree Classifier** (**83.33% verified test accuracy**).

---

## 🌟 Key Highlights & Features

1. **HR Analytics Dashboard**:
   - Executive KPI cards: Total Workforce (1,470), Attrition Rate (16.12%), Retained Headcount (1,233), Average Salary ($6,503/mo), and Average Tenure (7.0 years).
   - Department-wise Attrition Bar Chart (Sales, R&D, HR).
   - Job Satisfaction vs. Turnover Visualization (highlighting 2x higher turnover in low satisfaction tiers).
   - Overtime Impact Analysis (revealing that overtime workers experience **30.5% attrition** vs **10.4%** without overtime).
   - Job Role Flight Risk rankings.

2. **Interactive Attrition Predictor**:
   - Comprehensive employee profile form covering core attributes (Age, Department, Job Role, Monthly Income, Overtime, Job Satisfaction, Years at Company, Work-Life Balance) and advanced parameters (Commute distance, Experience, Promotions, Marital status).
   - **One-Click Quick Presets**: Test real-world scenarios immediately ("High Risk Overtime", "Low Risk Veteran", "Colab Test Sample #1061", "Moderate Risk Scientist").
   - **Real-Time Prediction Engine**: Connected to the Python Flask backend running Scikit-Learn `DecisionTreeClassifier`.
   - **Visual Risk Score Gauge**: 0% to 100% turnover probability meter with Low, Moderate, and High risk classifications.
   - **Explainability & Decision Path**: Extracts the exact decision tree branch rules evaluated for each prediction.
   - **HR Retention Interventions**: Automated recommendations tailored to identified risk drivers (e.g., workload rebalancing, compensation audit, mentorship mapping).

3. **Model Evaluation & Metrics**:
   - **Confirmed Test Accuracy**: **83.33%** (245 out of 294 test instances correctly classified on unseen test data).
   - **Interactive Confusion Matrix**:
     - **True Negatives (TN)**: 237 (Stayed correctly identified)
     - **False Positives (FP)**: 10 (False alarm)
     - **False Negatives (FN)**: 39 (Missed turnover)
     - **True Positives (TP)**: 8 (Attrition correctly flagged)
   - Complete Classification Report (Precision, Recall, F1-Score, Support for each class).
   - Horizontal Feature Importance Bar Chart highlighting the top Gini impurity reducers.

4. **About Project Documentation**:
   - Comprehensive writeup explaining project objectives, IBM dataset attributes, machine learning methodology, tools, and limitations.

5. **Ethical AI & Academic Disclaimer**:
   - Clearly positioned notice emphasizing that this project is an educational and research demonstration and must not be used as the sole basis for real-world employment decisions.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | React 19, Vite, Vanilla CSS Glassmorphism Design System, Lucide React Icons |
| **Visualizations** | Recharts, SVG Dynamic Risk Gauge |
| **Backend API** | Python 3.11, Flask, Flask-CORS |
| **Machine Learning** | Scikit-Learn (`DecisionTreeClassifier`, `max_depth=5, random_state=42`) |
| **Data Processing & Storage** | Pandas, NumPy, Joblib |
| **Dataset** | IBM HR Analytics Employee Attrition & Performance (`1,470 records, 35 features`) |

---

## 🔬 The Google Colab Connection & 83.33% Accuracy Verification

The model reproduces the exact training methodology from the **Google Colab project notebook**:

1. **Feature Preprocessing**:
   - Non-predictive metadata dropped: `['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']`.
   - Categorical columns transformed via `pd.get_dummies(X, drop_first=True)` into 44 numeric features.
2. **Train/Test Split**:
   - Stratified 80/20 train-test split (`test_size=0.2, random_state=42, stratify=y`).
   - Training records: **1,176** | Testing records: **294**.
3. **Model Configuration**:
   - `DecisionTreeClassifier(max_depth=5, random_state=42)`.
4. **Evaluation Verification**:
   - Test Accuracy: $\frac{237 + 8}{294} = \frac{245}{294} = \mathbf{83.33\%}$.
5. **Persistence**:
   - The trained model and feature schemas are saved to `backend/models/decision_tree_model.joblib` and `backend/models/model_metadata.json` using `joblib`.

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── data/
│   │   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Official IBM dataset
│   ├── models/
│   │   ├── decision_tree_model.joblib             # Trained model artifact
│   │   ├── model_metadata.json                   # Preprocessing schema & column mapping
│   │   ├── evaluation_results.json               # Exact test metrics & confusion matrix
│   │   └── dashboard_stats.json                  # Precalculated HR analytics metrics
│   ├── train.py                                  # Training script reproducing Colab run
│   ├── app.py                                    # Flask REST API server
│   ├── test_api.py                               # API test suite
│   └── requirements.txt                          # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx                        # Brand header, tabs, and API status pill
│   │   │   ├── DisclaimerBanner.jsx              # Educational demonstration banner
│   │   │   ├── KpiCard.jsx                       # Reusable KPI metric card
│   │   │   ├── RiskGauge.jsx                     # Animated SVG turnover risk gauge
│   │   │   ├── ConfusionMatrixView.jsx           # Interactive confusion matrix
│   │   │   └── Footer.jsx                        # Attribution footer
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx                     # HR Analytics dashboard with charts
│   │   │   ├── PredictAttrition.jsx              # Dynamic prediction form & results
│   │   │   ├── ModelEvaluation.jsx               # Accuracy (83.33%) & performance metrics
│   │   │   └── AboutProject.jsx                  # Project background & methodology
│   │   ├── services/
│   │   │   └── api.js                            # API client connecting to Flask
│   │   ├── App.jsx                               # Root application
│   │   └── index.css                             # Glassmorphism design system
│   ├── index.html                                # App shell with Google Fonts
│   └── package.json
├── run_backend.bat                               # Windows launcher for Flask API
├── run_frontend.bat                              # Windows launcher for Vite React
├── run_all.bat                                   # One-click launcher for full stack
└── README.md                                     # Project documentation
```

---

## 🚀 Step-by-Step Setup & Running Instructions

### Prerequisites
- **Python**: Version 3.10 or higher.
- **Node.js**: Version 18 or higher (with npm).

---

### Method 1: One-Click Quick Launch (Windows)
Double-click `run_all.bat` in the root project folder. It will:
1. Start the Flask backend server on `http://127.0.0.1:5000`.
2. Start the Vite React development server on `http://localhost:5173`.
3. Automatically open your default web browser to the application.

---

### Method 2: Manual Terminal Setup

#### Step 1: Start the Python Flask Backend
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Re-train and verify the model:
   ```bash
   python train.py
   ```
   *Expected output: `Evaluation Accuracy: 83.33%`, `Confusion Matrix: [[237, 10], [39, 8]]`.*
4. Start the Flask API server:
   ```bash
   python app.py
   ```
   *The backend will run at `http://127.0.0.1:5000`.*

#### Step 2: Start the React Frontend
1. Open a second terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Launch the development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to `http://localhost:5173/`.

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns service status, model loaded status, and verified accuracy (83.33%). |
| `GET` | `/api/dashboard` | Returns precomputed HR analytics KPIs and aggregated chart statistics. |
| `GET` | `/api/evaluation` | Returns confusion matrix, precision, recall, F1, and feature importances. |
| `GET` | `/api/schema` | Returns feature schema, allowable options, and numerical ranges. |
| `GET` | `/api/presets` | Returns curated sample employee profiles for quick testing. |
| `POST` | `/api/predict` | Accepts employee profile JSON and returns turnover prediction, probability, risk level, contributing factors, recommendations, and decision path. |
| `POST` | `/api/retrain` | Retrains the model from `backend/data/` and updates runtime artifacts. |

---

## ⚠️ Important Educational Disclaimer

This application was engineered solely as an academic capstone demonstration. Predictions are probabilistic research outputs from a Decision Tree Classifier trained on historical benchmark data. This software **is not intended and must not be utilized** as an automated decision-making tool for hiring, performance appraisal, promotion, compensation, disciplinary action, or termination of employment. Real-world talent management requires human empathy, individual context, and organizational ethics.
