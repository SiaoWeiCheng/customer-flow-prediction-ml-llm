# customer-flow-prediction-ml-llm
# 🏥 Customer Flow Prediction Using Machine Learning & Large Language Models (FYP)

> An end-to-end AI framework for predicting patient demand, optimizing clinic operations, and generating explainable staffing recommendations across a network of **98 private clinics**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange)

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
      ↓
Phase 1  → Strategic Profiling
      ↓
Phase 2  → Patient Demand Forecasting
      ↓
Phase 3  → Wait Time Simulation
      ↓
Phase 4  → Crisis Detection
      ↓
Phase 5  → Future Inference (7-Day Forecast)
      ↓
Gemini AI Assistant (Decision Support Layer)
```

---

# ⚙️ AI Pipeline

## Phase 1 — Strategic Profiling

- K-Means clustering
- Strategic 4-Quadrant Matrix
- Cold-start solution for new clinics

---

## Phase 2 — Demand Forecasting

- XGBoost/ARIMA/Prophet
- Walk-forward validation
- 7-day rolling prediction of patient arrivals

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

- Detect potential operational bottlenecks
- Identify high-risk periods before they occur
- Minimize missed critical cases while reducing alert fatigue

---

## Phase 5 — Future Inference

The trained framework generates a **7-day operational forecast** for each clinic, allowing managers to proactively plan resources before bottlenecks occur.

Predicted outputs include:

- 👥 Daily patient volume
- ⏱️ Expected average waiting time
- 👨‍⚕️ Recommended staff count

The future inference module provides clinic managers with an operational outlook, enabling proactive staffing decisions and better resource allocation across the clinic network.

---

# 🤖 AI Decision Support

To improve interpretability and usability, the framework integrates **Gemini 2.5 Flash** as an AI decision-support assistant.

Rather than generating predictions, the LLM interprets model outputs and presents them in a concise, manager-friendly format.

Capabilities include:

- 📋 Daily operational summaries
- 💬 Natural language data queries
- 🔍 SHAP-based root cause explanations
- 👨‍⚕️ AI-assisted staffing recommendations
- 📊 Human-readable reports for clinic managers

To ensure factual consistency, the assistant adopts a **Table-Augmented Generation (TAG)** approach with prompt sandboxing, preventing unsupported or hallucinated recommendations.

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Machine Learning | XGBoost, ARIMA, Prophet, Scikit-Learn |
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

## 🚀 How to Run Locally
1. Clone the repository: `git clone https://github.com/SiaoWeiCheng/customer-flow-prediction-ml-llm.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `streamlit run src/app.py`

---
