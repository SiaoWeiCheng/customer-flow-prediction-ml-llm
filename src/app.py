import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib 
import pandasai as pai
from pandasai_litellm import LiteLLM
from pandasai import Agent
import litellm

# Disable automatic retries
litellm.num_retries = 0
litellm.set_verbose = True

# =========================================================
# 1. PAGE CONFIGURATION & API SETUP
# =========================================================
st.set_page_config(
    page_title="Clinic AI Command Center",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API CONFIGURATION ---
GOOGLE_API_KEY = "xxx" # Replace xxx with your own API key
genai.configure(api_key=GOOGLE_API_KEY)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stAlert {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏥 Clinic AI Operations Command Center")
st.markdown("### *Predictive Analytics & Patient Flow Optimization System*")
st.markdown("---")

# =========================================================
# 2. HELPER FUNCTIONS (DATA & AI)
# =========================================================

@st.cache_data
def load_data():
    try:
        df_future = pd.read_csv("../data/next_week_forecast_final.csv")
        df_future["Date"] = pd.to_datetime(df_future["Date"])
    except FileNotFoundError:
        df_future = pd.DataFrame()

    try:
        df_history = pd.read_csv("../data/processed_daily_operations.csv")
        df_history["Date"] = pd.to_datetime(df_history["Date"])
        if "High_Wait_Risk" not in df_history.columns:
            df_history["High_Wait_Risk"] = (df_history["Avg_Wait_Time"] > 30).astype(int)
    except FileNotFoundError:
        df_history = pd.DataFrame()

    try:
        df_clusters = pd.read_csv("../data/clinic_clusters_final.csv")
        df_static = pd.read_csv("../data/clinic_static_profiles.csv")
        # Merge encoded clusters into history
        if not df_history.empty and not df_static.empty:
            df_history = pd.merge(
                df_history,
                df_static[["Name", "Cluster_Encoded"]],
                on="Name",
                how="left",
            )
            df_history["Cluster_Encoded"] = df_history["Cluster_Encoded"].fillna(-1)
    except FileNotFoundError:
        df_clusters = pd.DataFrame()

    return df_future, df_history, df_clusters


# =========================================================
# 🔥 THE FIX: LOADING INSTEAD OF RETRAINING
# =========================================================
@st.cache_resource
def load_interactive_models():
    """Loads the pre-trained Wait Time Simulator for live 'What-If' scenarios."""
    try:
        # Load the features you saved
        features_wait = joblib.load('../models/waittime_feature_columns.pkl')
        
        # Load the XGBoost model you saved
        xgb_wait = XGBRegressor()
        xgb_wait.load_model('../models/xgboost_waittime_simulator.json')
        
        return xgb_wait, features_wait
    except FileNotFoundError:
        st.error("❌ Models not found! Make sure 'xgboost_waittime_simulator.json' and 'waittime_feature_columns.pkl' are in the 'models/' folder.")
        return None, None


# --- AI LOGIC FUNCTIONS ---
def prepare_strategic_context(df_future, df_history, shap_model):
    if df_future.empty: return None
    df = df_future.copy()
    df['Date_Obj'] = pd.to_datetime(df['Date'])
    
    latest_date = df['Date_Obj'].max()
    week_start_cutoff = latest_date - pd.Timedelta(days=7)
    df = df[df['Date_Obj'] >= week_start_cutoff]

    start_date = df['Date_Obj'].min().strftime('%d %b %Y')
    end_date = df['Date_Obj'].max().strftime('%d %b %Y')
    
    target_list = df[df['Cluster_Name'].str.contains('Critical|Question|Star|Gem', case=False, na=False)]['Name'].unique().tolist()
    
    table_rows = []
    stale_count = 0

    def format_dates(date_series):
        if date_series.empty: return "-"
        return ", ".join([d.strftime('%d %b') for d in sorted(date_series.unique())])

    for clinic in target_list:
        c_data = df[df['Name'] == clinic]
        if c_data.empty: continue

        cluster_full = c_data['Cluster_Name'].iloc[0] if 'Cluster_Name' in c_data.columns else "Unknown"
        clinic_display_name = clinic
        if 'Data_Quality_Status' in c_data.columns and c_data['Data_Quality_Status'].astype(str).str.contains("STALE").any():
            clinic_display_name = f"{clinic} ⚠️"
            stale_count += 1

        if "Critical" in cluster_full: category = "🔴 Critical (Chronic)"
        elif "Question" in cluster_full: category = "❓ Question Mark"
        elif "Star" in cluster_full: category = "🌟 Star (Acute Alert)" 
        else: category = "💎 Gem (Acute Alert)" 

        crisis_df = c_data[c_data['Day_Status'] == 'Crisis Day']
        busy_df = c_data[c_data['Day_Status'] == 'Busy Day']
        
        max_val = crisis_df['Predicted_Wait_Time'].max() if not crisis_df.empty else 0
        max_wait_str = f"{max_val:.0f} min" if not crisis_df.empty else "-"
        
        # ---------------------------------------------------------
        # ⚙️ THE INVISIBLE OPTIMIZER: Run math in the background
        # ---------------------------------------------------------
        prescription = "Monitor closely."
        if not crisis_df.empty and shap_model is not None:
            worst_day_row = crisis_df.loc[crisis_df['Predicted_Wait_Time'].idxmax()]
            patients = worst_day_row['Total_Daily_Patients']
            original_staff = worst_day_row['Staff_Count']
            worst_date = worst_day_row['Date'].strftime('%Y-%m-%d')
            
            cluster_id = -1
            try:
                c_static = pd.read_csv('../data/clinic_static_profiles.csv')
                cluster_id = c_static[c_static['Name'] == clinic]['Cluster_Encoded'].iloc[0]
            except: pass
            
            hist_wait = 20.0
            if not df_history.empty and not df_history[df_history['Name'] == clinic].empty:
                hist_wait = df_history[df_history['Name'] == clinic]['Historical_Base_Wait'].iloc[0]

            SAFE_LIMIT, max_extra_staff, extra_staff_needed = 30.0, 5, 0
            current_wait = max_val
            
            while current_wait > SAFE_LIMIT and extra_staff_needed < max_extra_staff:
                extra_staff_needed += 1
                curr_staff = original_staff + extra_staff_needed
                
                # Using the loaded model features exactly as they were trained
                hypo = pd.DataFrame([{
                    'Total_Daily_Patients': patients, 'Staff_Count': curr_staff,
                    'Patient_Load_Ratio': patients / curr_staff, 'Load_Stress_Index': (patients / curr_staff)**2,
                    'Historical_Base_Wait': hist_wait, 'Cluster_Encoded': cluster_id, 
                    'IsWeekend': 1 if pd.to_datetime(worst_date).dayofweek >= 5 else 0, 'IsPublicHoliday': 0
                }]).astype(float)
                
                # Match column order to the loaded model
                try:
                    features_wait = joblib.load('../models/waittime_feature_columns.pkl')
                    hypo = hypo.reindex(columns=features_wait, fill_value=0)
                except: pass

                current_wait = shap_model.predict(hypo)[0]
            
            same_day_network = df[(df['Date_Obj'] == worst_day_row['Date_Obj']) & (df['Name'] != clinic) & (df['Staff_Count'] > 1)].copy()
            if current_wait <= SAFE_LIMIT and not same_day_network.empty:
                same_day_network['Load'] = same_day_network['Total_Daily_Patients'] / same_day_network['Staff_Count']
                best_donor = same_day_network.sort_values('Load').iloc[0]['Name']
                prescription = f"Require +{extra_staff_needed} staff. Pull from {best_donor}."
            elif current_wait <= SAFE_LIMIT:
                prescription = f"Require +{extra_staff_needed} staff (External)."
            else:
                prescription = "Max capacity. Reroute patients."

        crisis_dates = format_dates(crisis_df['Date_Obj'])
        busy_dates = format_dates(busy_df['Date_Obj'])
        
        if crisis_dates != "-" or busy_dates != "-":
            table_rows.append({
                "Name": clinic_display_name, "Category": category, "MaxWait": max_wait_str,
                "Crisis": crisis_dates, "Busy": busy_dates, "Prescription": prescription
            })
    
    if not table_rows:
        formatted_table = "No critical alerts this week."
    else:
        table_rows.sort(key=lambda x: 1 if "Critical" in x['Category'] else 2)
        formatted_table = "| Clinic | Status | Est. Max Wait | CRITICAL DATES | AI PRESCRIPTION |\n|---|---|---|---|---|\n"
        for r in table_rows:
            formatted_table += f"| {r['Name']} | {r['Category']} | {r['MaxWait']} | {r['Crisis']} | **{r['Prescription']}** |\n"

    return {
        'start': start_date, 'end': end_date,
        'crit_count': len([r for r in table_rows if "Critical" in r['Category']]),
        'quest_count': len([r for r in table_rows if "Question" in r['Category']]),
        'star_alert_count': len([r for r in table_rows if "Star" in r['Category']]),
        'stale_count': stale_count, 'schedule_table': formatted_table
    }

def generate_executive_prompt(ctx):
    stale_warning = f"\n    - **⚠️ DATA LATENCY WARNING:** {ctx['stale_count']} clinics are reporting delayed data." if ctx['stale_count'] > 0 else ""
    return f"""
    You are the AI Operations Director for a healthcare network. Write a 'Strategic Risk Brief' for the Regional Manager for the week of {ctx['start']} to {ctx['end']}.

    ### 1. SITUATION ANALYSIS
    - **Chronic Issues:** {ctx['crit_count'] + ctx['quest_count']} clinics flagged.
    - **Acute Alerts:** {ctx['star_alert_count']} high-performing clinics have specific "Crisis Days".{stale_warning}

    ### 2. OPERATIONAL SCHEDULE & AI PRESCRIPTIONS
    The XGBoost optimization engine has calculated the exact staff reallocation required. DO NOT invent your own solutions. Rely strictly on the 'AI PRESCRIPTION' column below:
    
    {ctx['schedule_table']}

    ### TASK:
    Write a professional executive summary (in Markdown):
    1.  **Prioritize:** Highlight clinics with the highest "Est. Max Wait".
    2.  **Action Plan:** Explicitly state the staff transfers recommended in the 'AI PRESCRIPTION' column (e.g., "Transfer 2 staff from Clinic X to Clinic Y").
    3.  **Output:** Print the provided table exactly at the bottom.
    """


# =========================================================
# 3. MAIN APP LOGIC
# =========================================================
df_future, df_history, df_clusters = load_data()

# 🔥 Use the loaded models, don't retrain!
shap_model, shap_features = load_interactive_models()

# Sidebar
st.sidebar.header("🔍 Control Panel")
all_clinics = sorted(df_history["Name"].unique()) if not df_history.empty else []
selected_clinic = st.sidebar.selectbox("Select Clinic:", options=["All"] + all_clinics)
app_mode = st.sidebar.radio(
    "Navigate to:",
    [
        "🔮 Next Week Forecast",
        "📜 Historical Analysis",
        "🌍 Strategic Clusters",
        "📝 AI Strategic Report",
        "💬 AI Data Assistant",
        "🛠️ AI Crisis Resolution",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status:**\n✅ Forecasting: Active\n✅ AI Reporting: Ready\n✅ Chat Agent: Online\n✅ XAI Engine: Loaded"
)

# Filter Data
if selected_clinic != "All":
    future_view = df_future[df_future["Name"] == selected_clinic]
    history_view = df_history[df_history["Name"] == selected_clinic]
else:
    future_view = df_future
    history_view = df_history

# --- TAB 1: FORECAST ---
if app_mode == "🔮 Next Week Forecast":
    st.header(f"🔮 Forecast: Next 7 Days ({selected_clinic})")
    if future_view.empty:
        st.warning("No forecast data available.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Projected Volume", f"{future_view['Total_Daily_Patients'].sum():,.0f}")
        col2.metric("Avg Predicted Wait", f"{future_view['Predicted_Wait_Time'].mean():.1f} min", delta_color="inverse")
        col3.metric("Crisis Days", f"{future_view[future_view['Day_Status'] == 'Crisis Day'].shape[0]}", delta_color="inverse")
        col4.metric("Avg Staff", f"{future_view['Staff_Count'].mean():.1f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=future_view["Date"], y=future_view["Total_Daily_Patients"], name="Volume", marker_color="#3b8ed0", opacity=0.6))
        fig.add_trace(go.Scatter(x=future_view["Date"], y=future_view["Predicted_Wait_Time"], name="Wait Time", yaxis="y2", line=dict(color="#e04f5f", width=3)))
        fig.add_trace(go.Scatter(x=future_view["Date"], y=[30] * len(future_view), name="Limit", yaxis="y2", line=dict(color="red", dash="dot")))
        fig.update_layout(yaxis=dict(title="Patients"), yaxis2=dict(title="Minutes", overlaying="y", side="right"), title="Volume vs Wait Time")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(future_view[["Date", "Name", "Total_Daily_Patients", "Staff_Count", "Predicted_Wait_Time", "Day_Status"]], use_container_width=True)

# --- TAB 2: HISTORY ---
elif app_mode == "📜 Historical Analysis":
    st.header(f"📜 Historical Performance ({selected_clinic})")
    metric = st.selectbox("Select Metric:", ["Avg_Wait_Time", "Total_Daily_Patients", "Patient_Load_Ratio"])
    st.plotly_chart(px.line(history_view, x="Date", y=metric, color="Name" if selected_clinic == "All" else None), use_container_width=True)

# --- TAB 3: CLUSTERS ---
elif app_mode == "🌍 Strategic Clusters":
    st.header("🌍 Strategic Clinic Segmentation")
    if not df_clusters.empty:
        fig = px.scatter(df_clusters, x="Avg_Volume", y="Avg_Wait", color="Cluster_Label", hover_name="Name", title="Strategic Matrix")
        fig.add_vline(x=df_clusters["Avg_Volume"].mean(), line_dash="dash")
        fig.add_hline(y=df_clusters["Avg_Wait"].mean(), line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Cluster data missing.")

# --- TAB 4: AI REPORT ---
elif app_mode == "📝 AI Strategic Report":
    st.header("📝 AI Executive Report Generation")
    st.markdown("Generate a natural language briefing for the Regional Manager based on the latest forecasts.")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Generate Report", type="primary"):
            if df_future.empty:
                st.error("Cannot generate report: No forecast data found.")
            else:
                with st.spinner("🤖 AI is analyzing network risks..."):
                    ctx = prepare_strategic_context(df_future, df_history, shap_model)
                    if ctx:
                        prompt = generate_executive_prompt(ctx)
                        try:
                            model = genai.GenerativeModel("gemini-2.5-flash")
                            response = model.generate_content(prompt)
                            report_text = response.text
                            st.session_state["generated_report"] = report_text
                            os.makedirs("../results", exist_ok=True)
                            with open("../results/Regional_Strategy_Report.md", "w", encoding="utf-8") as f:
                                f.write(report_text)
                            st.success("Report generated successfully!")
                        except Exception as e:
                            st.error(f"AI Error: {e}")
                    else:
                        st.warning("No critical risks found to report.")

    with col2:
        if "generated_report" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["generated_report"])
            st.download_button(label="📥 Download Report as Markdown", data=st.session_state["generated_report"], file_name="Regional_Strategy_Report.md", mime="text/markdown")
        else:
            st.info("Click 'Generate Report' to create a new strategic brief.")

# --- TAB 5: AI DATA ASSISTANT ---
elif app_mode == "💬 AI Data Assistant":
    st.header("💬 AI Data Assistant (Tabular RAG)")
    st.markdown("Ask natural language questions about your historical clinic operations data.")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()

    try:
        llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY, max_retries=0)
        pai.config.set({"llm": llm, "save_charts": True})
        if not df_history.empty:
            agent = Agent(df_history)
        else:
            st.error("Historical data is empty.")
            st.stop()
    except Exception as e:
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "RateLimitError" in error_text:
            st.error("🚫 Gemini quota exceeded while initializing the AI assistant.")
        else:
            st.error(f"Failed to initialize AI Agent: {e}")
        st.stop()

    if "chat_messages" not in st.session_state or len(st.session_state.chat_messages) == 0:
        st.session_state.chat_messages = [{"role": "assistant", "type": "text", "content": "Hello! I am your AI Operations Analyst. What would you like to know?"}]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text": st.markdown(msg["content"])
            elif msg["type"] == "code": st.code(msg["content"], language="text")
            elif msg["type"] == "dataframe": st.dataframe(msg["content"], use_container_width=True)
            elif msg["type"] == "image": st.image(msg["content"])

    if user_input := st.chat_input("Ask a question about your data..."):
        st.session_state.chat_messages.append({"role": "user", "type": "text", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Writing Pandas code and analyzing data..."):
                try:
                    response = agent.chat(user_input)
                    if isinstance(response, (pd.DataFrame, pd.Series)) or type(response).__name__ in ["SmartDataframe", "DataFrame"]:
                        st.dataframe(response, use_container_width=True)
                        st.session_state.chat_messages.append({"role": "assistant", "type": "dataframe", "content": response})
                    elif isinstance(response, str) and (".png" in response or ".jpg" in response or "exports" in response):
                        clean_path = response.strip()
                        if os.path.exists(clean_path):
                            st.image(clean_path)
                            st.session_state.chat_messages.append({"role": "assistant", "type": "image", "content": clean_path})
                        else:
                            st.error(f"Chart generated but file not found at: {clean_path}")
                    else:
                        response_str = str(response)
                        if "\n" in response_str and "  " in response_str:
                            st.code(response_str, language="text")
                            st.session_state.chat_messages.append({"role": "assistant", "type": "code", "content": response_str})
                        else:
                            formatted_str = response_str.replace("\n", "  \n")
                            st.markdown(formatted_str)
                            st.session_state.chat_messages.append({"role": "assistant", "type": "text", "content": formatted_str})
                except Exception as e:
                    error_text = str(e)
                    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower() or "RateLimitError" in error_text:
                        error_msg = "🚫 Gemini quota exceeded.\nThe AI assistant has stopped to prevent additional retries."
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "type": "text", "content": error_msg})
                        st.stop()
                    error_msg = f"Sorry, I ran into an error:\n\n{error_text}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "type": "text", "content": error_msg})

# --- TAB 6: AI CRISIS RESOLUTION (SHAP + OPTIMIZER) ---
elif app_mode == "🛠️ AI Crisis Resolution":
    st.header("🛠️ AI Crisis Resolution Center")
    st.markdown("Diagnose the root cause of forecasted bottlenecks (SHAP) and automatically generate a staff reallocation strategy.")

    if shap_model is None:
        st.error("Model could not be loaded. Check your data files.")
    else:
        # Filter for Crisis Days
        crisis_days = df_future[df_future["Day_Status"] == "Crisis Day"]

        if crisis_days.empty:
            st.success("🎉 Network is fully optimized! No crises forecasted for next week.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_clinic = st.selectbox("Select Forecasted Crisis:", crisis_days["Name"].unique())
            with col2:
                clinic_dates = crisis_days[crisis_days["Name"] == target_clinic]["Date"].dt.strftime("%Y-%m-%d").unique()
                selected_date = st.selectbox("Select Date:", clinic_dates)

            st.markdown("---")

            target_row = crisis_days[(crisis_days["Name"] == target_clinic) & (crisis_days["Date"].dt.strftime("%Y-%m-%d") == selected_date)].iloc[0]
            patients = target_row["Total_Daily_Patients"]
            original_staff = target_row["Staff_Count"]
            predicted_wait = target_row["Predicted_Wait_Time"]

            st.warning(f"🚨 **CRISIS ALERT:** {target_clinic} on {selected_date} | Projected Wait: {predicted_wait:.1f} mins | Volume: {patients} | Staff: {original_staff}")

            if st.button("🧠 Analyze Root Cause & Generate Solution", type="primary"):
                try:
                    c_static = pd.read_csv("../data/clinic_static_profiles.csv")
                    cluster_id = c_static[c_static["Name"] == target_clinic]["Cluster_Encoded"].iloc[0]
                except:
                    cluster_id = -1

                hist_wait = 20.0
                if not df_history.empty:
                    c_hist = df_history[df_history["Name"] == target_clinic]
                    if not c_hist.empty:
                        hist_wait = c_hist["Historical_Base_Wait"].iloc[0]

                target_features = pd.DataFrame([{
                    "Total_Daily_Patients": patients, "Staff_Count": original_staff,
                    "Patient_Load_Ratio": patients / original_staff, "Load_Stress_Index": (patients / original_staff) ** 2,
                    "Historical_Base_Wait": hist_wait, "Cluster_Encoded": cluster_id,
                    "IsWeekend": 1 if pd.to_datetime(selected_date).dayofweek >= 5 else 0, "IsPublicHoliday": 0,
                }]).astype(float)
                
                # Enforce the exact order from your extracted pkl
                target_features = target_features.reindex(columns=shap_features, fill_value=0)

                col_diag, col_sol = st.columns([1.2, 1])

                # PART 1: DIAGNOSIS (SHAP)
                with col_diag:
                    st.subheader("1. Diagnosis (Why is this happening?)")
                    with st.spinner("Calculating SHAP values..."):
                        
                        # 1. One-hot encode the history exactly like we did in training
                        hist_encoded = df_history.copy()
                        if 'State' in hist_encoded.columns and 'District' in hist_encoded.columns:
                            # Added dtype=float to force 1.0/0.0 instead of True/False booleans
                            hist_encoded = pd.get_dummies(hist_encoded, columns=['State', 'District'], dtype=float)
                            
                        # 2. Auto-fill any missing geographic columns with 0
                        hist_encoded = hist_encoded.reindex(columns=shap_features + ["Avg_Wait_Time"], fill_value=0)
                        
                        # 3. Now it is perfectly safe to drop NaNs and sample!
                        train_df = hist_encoded.dropna(subset=shap_features + ["Avg_Wait_Time"])
                        
                        # Fallback just in case the history is empty
                        if train_df.empty:
                            background_data = pd.DataFrame([0]*len(shap_features), index=shap_features).T
                        else:
                            background_data = train_df[shap_features].sample(n=min(100, len(train_df)), random_state=42)
                        
                        # 4. 🔥 THE FIX: Force both background and target to float so SHAP math doesn't crash
                        background_data = background_data.astype(float)
                        target_features = target_features.astype(float)
                        
                        try:
                            explainer = shap.Explainer(shap_model.predict, background_data)
                            shap_values = explainer(target_features)
                            
                            fig, ax = plt.subplots(figsize=(8, 5))
                            shap.plots.waterfall(shap_values[0], show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                        except Exception as e:
                            st.error(f"Error generating explanation: {e}")

                # PART 2: SOLUTION (OPTIMIZER)
                with col_sol:
                    st.subheader("2. Solution (Staff Reallocation)")
                    with st.spinner("Simulating network optimization..."):
                        SAFE_LIMIT = 30.0
                        max_extra_staff = 5
                        current_staff = original_staff
                        extra_staff_needed = 0
                        current_wait = predicted_wait

                        while current_wait > SAFE_LIMIT and extra_staff_needed < max_extra_staff:
                            extra_staff_needed += 1
                            current_staff = original_staff + extra_staff_needed
                            hypo_scenario = pd.DataFrame([{
                                "Total_Daily_Patients": patients, "Staff_Count": current_staff,
                                "Patient_Load_Ratio": patients / current_staff, "Load_Stress_Index": (patients / current_staff) ** 2,
                                "Historical_Base_Wait": hist_wait, "Cluster_Encoded": cluster_id,
                                "IsWeekend": 1 if pd.to_datetime(selected_date).dayofweek >= 5 else 0, "IsPublicHoliday": 0,
                            }]).astype(float)
                            
                            # Enforce exact order
                            hypo_scenario = hypo_scenario.reindex(columns=shap_features, fill_value=0)
                            current_wait = shap_model.predict(hypo_scenario)[0]

                        same_day_network = df_future[
                            (df_future["Date"].dt.strftime("%Y-%m-%d") == selected_date)
                            & (df_future["Name"] != target_clinic) & (df_future["Staff_Count"] > 1)
                        ].copy()

                        same_day_network["Patient_Load_Ratio"] = same_day_network["Total_Daily_Patients"] / same_day_network["Staff_Count"]
                        donor_pool = same_day_network.sort_values("Patient_Load_Ratio", ascending=True)

                        if current_wait <= SAFE_LIMIT:
                            st.success(f"**Action Required:** Transfer **+{extra_staff_needed} Float Staff** to {target_clinic}.")
                            st.info(f"📉 **Impact:** This drops the predicted wait time from {predicted_wait:.1f} to **{current_wait:.1f} minutes**.")

                            if not donor_pool.empty:
                                best_donor = donor_pool.iloc[0]["Name"]
                                donor_ratio = donor_pool.iloc[0]["Patient_Load_Ratio"]
                                donor_staff = donor_pool.iloc[0]["Staff_Count"]
                                st.markdown(f"🔄 **Donor Clinic:** Pull from **{best_donor}**.")
                                st.markdown(f"*They have {donor_staff} staff and only {donor_ratio:.1f} patients per doctor. They will remain fully operational.*")
                            else:
                                st.warning("⚠️ No safe donor clinic found. Use external float pool.")
                        else:
                            st.error(f"⚠️ **SEVERE SHORTAGE:** Cannot resolve with +{max_extra_staff} staff. Reroute patients immediately.")