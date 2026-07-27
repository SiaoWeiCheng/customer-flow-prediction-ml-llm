# customer-flow-prediction-ml-llm
# 🏥 Customer Flow Prediction Using Machine Learning & Large Language Models (FYP)

> An end-to-end AI framework for predicting patient demand, optimizing clinic operations, and generating explainable staffing recommendations across a network of **98 private clinics**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Healthcare providers often struggle to accurately predict patient demand, especially for **new clinics with little or no historical data**.

Traditional forecasting models also suffer from:

- ❌ Cold-start problems for new facilities
- ❌ Poor generalization across different clinic profiles
- ❌ Black-box predictions that managers cannot easily interpret

This project proposes a **5-phase AI framework** that combines **Machine Learning**, **Explainable AI (XAI)**, and **Large Language Models (LLMs)** to deliver accurate forecasts together with human-readable operational recommendations.

---

## ✨ Key Features

- 📈 Patient arrival forecasting
- ⏱️ Wait-time prediction using physics-informed machine learning
- 🏥 Automatic clinic profiling with K-Means clustering
- 🔍 SHAP Explainable AI diagnostics
- 🤖 Gemini-powered AI assistant
- 📊 Interactive Streamlit dashboard
- 📋 AI-generated staffing recommendations

---

# 🏗️ System Architecture

> *(Insert your architecture diagram here)*

```text
98 Clinics
      │
      ▼
K-Means Strategic Clustering
      │
      ▼
Patient Demand Forecasting
      │
      ▼
Physics-Constrained Wait Time Simulation
      │
      ▼
High-Risk Crisis Detection
      │
      ▼
SHAP Explainability
      │
      ▼
Gemini AI Assistant
      │
      ▼
Staff Reallocation Recommendation
```

---

# ⚙️ AI Pipeline

## Phase 1 — Strategic Profiling

- K-Means clustering
- Strategic 4-Quadrant Matrix
- Cold-start solution for new clinics

---

## Phase 2 — Demand Forecasting

- XGBoost Regressor
- Walk-forward validation
- 7-day rolling prediction

---

## Phase 3 — Wait Time Simulation

Physics-informed XGBoost model with monotonic constraints to ensure logically consistent predictions.

Example:

✅ More staff → Shorter waiting time

❌ More staff → Longer waiting time

---

## Phase 4 — Crisis Detection

High-sensitivity classifier optimized using an asymmetric decision threshold.

Objectives:

- Detect operational bottlenecks early
- Reduce missed critical cases
- Minimize alert fatigue

---

## Phase 5 — AI Decision Support

A Gemini 2.5 Flash agent converts technical model outputs into natural language reports for clinic managers.

The assistant provides:

- Daily operational summaries
- Staffing recommendations
- Root cause explanations
- Resource allocation suggestions

---

# 🧠 Explainable AI (XAI)

Instead of producing black-box predictions, the framework uses **SHAP Waterfall Analysis** to explain:

- Which features increased waiting time
- Which variables reduced waiting time
- Why a clinic is predicted to become overloaded

This allows healthcare managers to trust and validate AI decisions.

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Machine Learning | XGBoost, Scikit-Learn |
| Clustering | K-Means |
| Explainable AI | SHAP |
| LLM | Gemini 2.5 Flash |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |

---

# 📊 Results

| Task | Performance |
|------|-------------|
| Patient Demand Forecasting | RMSE = **6.85** |
| Wait Time Prediction | MAE = **2.41 minutes** |
| Crisis Detection | Recall = **87%** |
| ROC-AUC | **0.95** |

### Highlights

- ✅ Outperformed baseline ARIMA models
- ✅ Outperformed Prophet forecasting
- ✅ Cold-start prediction for new clinics
- ✅ Explainable AI with SHAP
- ✅ Zero-hallucination LLM integration

---

# 💻 Dashboard

The Streamlit dashboard provides:

- 🔮 Next-week forecasting
- 📜 Historical performance analysis
- 🌍 Strategic cluster visualization
- 📈 Interactive operational dashboard
- 💬 AI Data Assistant
- 🛠 AI Crisis Resolution

---

# 📂 Repository Structure

```text
customer-flow-prediction-ml-llm
│
├── data/
├── notebooks/
├── src/
├── models/
├── images/
├── results/
├── requirements.txt
└── README.md
```

---

# 🚀 Future Improvements

- Real-time IoT integration
- Multi-hospital deployment
- Reinforcement learning for staff scheduling
- Cloud deployment with Docker
- MLOps pipeline automation

---

# 👨‍💻 Author

**Siao Wei Cheng**

Bachelor of Computer Science (Data Science)

Multimedia University (MMU)

Interested in:

- Machine Learning
- Data Science
- Artificial Intelligence
- Explainable AI
- LLM Applications

---

## ⭐ If you found this project interesting, please consider giving it a star.
