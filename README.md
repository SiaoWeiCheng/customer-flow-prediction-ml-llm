# customer-flow-prediction-ml-llm
Customer Flow Prediction using Machine Learning and Large Language Models (Final Year Project)

📌 Project Overview
This repository contains a data-driven, sequential AI framework designed to optimize patient flow and resource allocation across a network of 98 private clinics.

Existing healthcare predictive models often suffer from a "cold start" problem for new facilities, fail to generalize across diverse clinic profiles, and operate as "black boxes" that clinic managers struggle to trust. This project solves these limitations by combining K-Means clustering, physics-informed XGBoost forecasting, and a Gemini-powered LLM agent to transform complex numerical predictions into safe, actionable staff reallocation strategies.

🚀 Key Architectural Contributions
Overcoming the "Cold Start" Problem (Transfer Learning): Engineered a K-Means clustering pipeline to group the 98 clinics into a Strategic 4-Quadrant Matrix. New clinics with zero historical data are assigned an operational profile, allowing the system to accurately predict demand on Day 1.

Physics-Informed Monotonic Constraints: Hard-coded queuing theory principles into the XGBoost wait-time simulator. The model is mathematically restricted from making illogical predictions (e.g., forecasting longer wait times when staff capacity increases).

Prescriptive Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) Waterfall diagnostics to isolate the root cause of predicted bottlenecks.

Zero-Hallucination LLM Integration: Developed an AI Data Assistant using Gemini 2.5 Flash and Table-Augmented Generation (TAG). Enforced strict Prompt Sandboxing to guarantee the LLM acts purely as a prescriptive data formatter with a 100% Factual Consistency Rate.

🛠️ Technology Stack
Machine Learning: XGBoost (Regressor & Classifier), Scikit-Learn, K-Means Clustering, SHAP

Large Language Models: Gemini 2.5 Flash API (Optimized for massive context window and low latency)

Data Processing: Python, Pandas, NumPy

Frontend Dashboard: Streamlit

⚙️ The 5-Phase Sequential Pipeline
Strategic Profiling: K-Means stratification of network dynamics.

Demand Forecasting: 7-day rolling walk-forward validation for patient arrivals.

Wait Time Simulation: Physics-constrained simulation of patient flow.

Crisis Detection: High-sensitivity classification optimized via an Asymmetric Cost Threshold.

Agentic Strategy Generation: LLM-driven reporting translating SHAP diagnostics into staffing solutions.

📊 Evaluation & Results
The framework was rigorously evaluated on historical clinic data:

Wait Time Simulation: Achieved a Mean Absolute Error (MAE) of 2.41 minutes.

Demand Forecasting: Achieved an average RMSE of 6.85, significantly outperforming baseline ARIMA and Prophet models.

High-Risk Crisis Detection: Achieved an 87% Recall and 0.95 ROC-AUC. The operating point was mathematically optimized to a 0.368 threshold using the Precision-Recall curve to prioritize patient safety without causing severe alert fatigue.

💻 Dashboard Interface
The system features a live Streamlit application serving as the clinic manager's control panel:

🔮 Next Week Forecast: View simulated future demand and wait times.

📜 Historical Analysis: Compare AI predictions against ground-truth baselines.

🌍 Strategic Clusters: View the operational topology of the 98-clinic network.

📝 AI Strategic Report: Generate automated morning briefings aggregating 7-day forecasts.

💬 AI Data Assistant: Query complex operational metrics using natural language.

🛠️ AI Crisis Resolution: View flagged bottlenecks, SHAP root-cause analysis, and AI-generated staff reallocation strategies.
