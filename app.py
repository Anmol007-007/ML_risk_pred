import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Workforce Attrition Predictor | Palo Alto Networks",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 20px;
        color: #fecaca;
    }
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(21, 128, 61, 0.25) 100%);
        border: 1px solid #22c55e;
        border-radius: 14px;
        padding: 20px;
        color: #bbf7d0;
    }
    .risk-banner-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%);
        border: 1px solid #f59e0b;
        border-radius: 14px;
        padding: 20px;
        color: #fde68a;
    }
    
    .recommendation-pill {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        model = joblib.load("best_model.pkl")
        return preprocessor, model, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_dataset():
    csv_path = "Palo Alto Networks.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def engineer_features(df):
    data = df.copy()
    data['Income_to_Experience'] = data['MonthlyIncome'] / (data['TotalWorkingYears'] + 1)
    data['Promotion_Delay_Flag'] = (data['YearsSinceLastPromotion'] > 3).astype(int)
    data['Engagement_Composite'] = (
        data['JobInvolvement'] + 
        data['JobSatisfaction'] + 
        data['EnvironmentSatisfaction'] + 
        data['RelationshipSatisfaction']
    ) / 4.0
    data['Workload_Stress_Flag'] = ((data['OverTime'] == 'Yes') & (data['WorkLifeBalance'] <= 2)).astype(int)
    return data

preprocessor, model, error_msg = load_models()
dataset_df = load_dataset()

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🚀 Workforce Retention & Attrition Intelligence</div>
    <div class="hero-subtitle">Predict employee turnover risk with XGBoost machine learning and uncover proactive retention strategies.</div>
</div>
""", unsafe_allow_html=True)

if error_msg:
    st.error(f"⚠️ Error loading ML models: {error_msg}")
    st.stop()

st.sidebar.image("https://img.icons8.com/isometric/100/conference-call.png", width=70)
st.sidebar.title("Attrition AI Suite")
menu = st.sidebar.radio(
    "Navigation",
    ["🔮 Individual Risk Predictor", "📂 Batch Employee Analysis", "📊 Workforce EDA & Trends", "🧠 Model Architecture & Importance"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Specs**:
- 🏷️ **Target**: Employee Attrition (0: Stays, 1: Leaves)
- ⚙️ **Classifier**: XGBoost (Extreme Gradient Boosting)
- 📐 **Preprocessing**: StandardScaler + OneHotEncoder
- 🧩 **Engineered Features**: 4 Custom HR Metrics
""")

if menu == "🔮 Individual Risk Predictor":
    st.subheader("Employee Profile & Real-time Risk Assessment")
    
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns([1, 1, 1, 2])
    with col_preset1:
        if st.button("🚨 Load High-Risk Profile", width='stretch'):
            st.session_state['preset'] = {
                'Age': 29, 'BusinessTravel': 'Travel_Frequently', 'DailyRate': 450,
                'Department': 'Sales', 'DistanceFromHome': 25, 'Education': 2,
                'EducationField': 'Marketing', 'EnvironmentSatisfaction': 1, 'Gender': 'Female',
                'HourlyRate': 45, 'JobInvolvement': 1, 'JobLevel': 1,
                'JobRole': 'Sales Representative', 'JobSatisfaction': 1, 'MaritalStatus': 'Single',
                'MonthlyIncome': 2300, 'MonthlyRate': 12000, 'NumCompaniesWorked': 6,
                'OverTime': 'Yes', 'PercentSalaryHike': 11, 'PerformanceRating': 3,
                'RelationshipSatisfaction': 1, 'StockOptionLevel': 0, 'TotalWorkingYears': 3,
                'TrainingTimesLastYear': 1, 'WorkLifeBalance': 1, 'YearsAtCompany': 2,
                'YearsInCurrentRole': 1, 'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 0
            }
    with col_preset2:
        if st.button("🌟 Load High-Retention Profile", width='stretch'):
            st.session_state['preset'] = {
                'Age': 44, 'BusinessTravel': 'Non-Travel', 'DailyRate': 1100,
                'Department': 'Research & Development', 'DistanceFromHome': 3, 'Education': 4,
                'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': 4, 'Gender': 'Male',
                'HourlyRate': 85, 'JobInvolvement': 4, 'JobLevel': 3,
                'JobRole': 'Healthcare Representative', 'JobSatisfaction': 4, 'MaritalStatus': 'Married',
                'MonthlyIncome': 10500, 'MonthlyRate': 18000, 'NumCompaniesWorked': 1,
                'OverTime': 'No', 'PercentSalaryHike': 18, 'PerformanceRating': 3,
                'RelationshipSatisfaction': 4, 'StockOptionLevel': 2, 'TotalWorkingYears': 16,
                'TrainingTimesLastYear': 4, 'WorkLifeBalance': 3, 'YearsAtCompany': 10,
                'YearsInCurrentRole': 7, 'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 7
            }
    with col_preset3:
        if st.button("🔄 Reset Defaults", width='stretch'):
            st.session_state.pop('preset', None)
            
    p = st.session_state.get('preset', {})
    
    with st.form("employee_form"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Personal & Demographics", 
            "💼 Role & Department", 
            "💰 Compensation & Schedule", 
            "⭐ Satisfaction & Culture"
        ])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 18, 60, p.get('Age', 35))
                gender = st.selectbox("Gender", ["Female", "Male"], index=0 if p.get('Gender') == 'Female' else 1)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], 
                                              index=["Single", "Married", "Divorced"].index(p.get('MaritalStatus', 'Married')))
            with col2:
                distance = st.slider("Distance From Home (miles)", 1, 30, p.get('DistanceFromHome', 8))
                education = st.select_slider("Education Level", options=[1, 2, 3, 4, 5], value=p.get('Education', 3),
                                             format_func=lambda x: {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}[x])
            with col3:
                education_field = st.selectbox("Education Field", 
                                               ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
                                               index=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"].index(p.get('EducationField', 'Life Sciences')))
                num_companies = st.slider("Number of Companies Worked At", 0, 9, p.get('NumCompaniesWorked', 2))

        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"],
                                          index=["Research & Development", "Sales", "Human Resources"].index(p.get('Department', 'Research & Development')))
                job_role = st.selectbox("Job Role", [
                    "Sales Executive", "Research Scientist", "Laboratory Technician", 
                    "Manufacturing Director", "Healthcare Representative", "Manager", 
                    "Sales Representative", "Research Director", "Human Resources"
                ], index=["Sales Executive", "Research Scientist", "Laboratory Technician", 
                          "Manufacturing Director", "Healthcare Representative", "Manager", 
                          "Sales Representative", "Research Director", "Human Resources"].index(p.get('JobRole', 'Research Scientist')))
                job_level = st.slider("Job Level", 1, 5, p.get('JobLevel', 2))
            with col2:
                total_working_years = st.slider("Total Working Experience (Years)", 0, 40, p.get('TotalWorkingYears', 10))
                years_at_company = st.slider("Years at Current Company", 0, 40, p.get('YearsAtCompany', 6))
                years_in_role = st.slider("Years in Current Role", 0, 18, p.get('YearsInCurrentRole', 4))
            with col3:
                years_since_promo = st.slider("Years Since Last Promotion", 0, 15, p.get('YearsSinceLastPromotion', 1))
                years_with_mgr = st.slider("Years With Current Manager", 0, 17, p.get('YearsWithCurrManager', 3))
                training_times = st.slider("Training Sessions Last Year", 0, 6, p.get('TrainingTimesLastYear', 3))

        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=p.get('MonthlyIncome', 6500), step=250)
                percent_hike = st.slider("Last Salary Hike (%)", 11, 25, p.get('PercentSalaryHike', 15))
                stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=p.get('StockOptionLevel', 1))
            with col2:
                overtime = st.radio("OverTime Hours?", ["No", "Yes"], index=1 if p.get('OverTime') == 'Yes' else 0, horizontal=True)
                business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                               index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(p.get('BusinessTravel', 'Travel_Rarely')))
            with col3:
                daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=p.get('DailyRate', 800), step=50)
                hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=p.get('HourlyRate', 65), step=5)
                monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=p.get('MonthlyRate', 14000), step=500)

        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                job_satisfaction = st.select_slider("Job Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], value=p.get('JobSatisfaction', 3))
                env_satisfaction = st.select_slider("Environment Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], value=p.get('EnvironmentSatisfaction', 3))
                relationship_sat = st.select_slider("Relationship Satisfaction (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], value=p.get('RelationshipSatisfaction', 3))
            with col2:
                job_involvement = st.select_slider("Job Involvement (1 = Low, 4 = Very High)", options=[1, 2, 3, 4], value=p.get('JobInvolvement', 3))
                work_life_balance = st.select_slider("Work-Life Balance (1 = Bad, 4 = Best)", options=[1, 2, 3, 4], value=p.get('WorkLifeBalance', 3))
                perf_rating = st.select_slider("Performance Rating", options=[3, 4], value=p.get('PerformanceRating', 3),
                                               format_func=lambda x: {3: "3 - Excellent", 4: "4 - Outstanding"}[x])

        submitted = st.form_submit_state = st.form_submit_button("⚡ Run Attrition Risk Analysis", width='stretch', type="primary")

    if submitted:
        input_dict = {
            'Age': age,
            'BusinessTravel': business_travel,
            'DailyRate': daily_rate,
            'Department': department,
            'DistanceFromHome': distance,
            'Education': education,
            'EducationField': education_field,
            'EnvironmentSatisfaction': env_satisfaction,
            'Gender': gender,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobRole': job_role,
            'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies,
            'OverTime': overtime,
            'PercentSalaryHike': percent_hike,
            'PerformanceRating': perf_rating,
            'RelationshipSatisfaction': relationship_sat,
            'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promo,
            'YearsWithCurrManager': years_with_mgr
        }
        
        input_df = pd.DataFrame([input_dict])
        engineered_df = engineer_features(input_df)
        
        try:
            X_trans = preprocessor.transform(engineered_df)
            prediction = model.predict(X_trans)[0]
            probabilities = model.predict_proba(X_trans)[0]
            attrition_prob = float(probabilities[1]) * 100
            retention_prob = float(probabilities[0]) * 100
            
            st.markdown("---")
            st.subheader("🎯 Risk Evaluation Results")
            
            col_gauge, col_details = st.columns([1.2, 1.8])
            
            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=attrition_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "<b>Turnover Risk Score</b>", 'font': {'size': 20, 'color': '#f8fafc'}},
                    number={'suffix': "%", 'font': {'size': 36, 'color': '#f8fafc'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#ef4444" if attrition_prob > 60 else ("#f59e0b" if attrition_prob > 30 else "#22c55e"), 'thickness': 0.28},
                        'bgcolor': "rgba(30, 41, 59, 0.5)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255, 255, 255, 0.1)",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.25)'},
                            {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.25)'},
                            {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                        ],
                        'threshold': {
                            'line': {'color': "#ef4444", 'width': 3},
                            'thickness': 0.8,
                            'value': 60
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'},
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gauge, width='stretch')
                
            with col_details:
                if attrition_prob >= 60:
                    st.markdown(f"""
                    <div class="risk-banner-high">
                        <h3 style="margin-top:0; color:#ef4444;">🚨 High Attrition Risk Detected</h3>
                        <p>This employee exhibits a <b>{attrition_prob:.1f}% probability</b> of leaving the organization within the upcoming quarter.</p>
                        <p><strong>Primary Vulnerabilities:</strong> Check Overtime demands, satisfaction indicators, and compensation trajectory.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif attrition_prob >= 30:
                    st.markdown(f"""
                    <div class="risk-banner-med">
                        <h3 style="margin-top:0; color:#f59e0b;">⚠️ Moderate Turnover Risk</h3>
                        <p>This employee shows a <b>{attrition_prob:.1f}% probability</b> of voluntary turnover. Targeted engagement conversations are recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-banner-low">
                        <h3 style="margin-top:0; color:#22c55e;">✅ High Retention / Stable</h3>
                        <p>This employee has a high retention probability of <b>{retention_prob:.1f}%</b> (Turnover risk is only {attrition_prob:.1f}%).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("#### 🔍 Engineered Feature Diagnostics")
                col_ef1, col_ef2, col_ef3, col_ef4 = st.columns(4)
                with col_ef1:
                    inc_exp = engineered_df['Income_to_Experience'].iloc[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Income / Exp</div>
                        <div class="metric-value" style="font-size:1.3rem;">${inc_exp:.0f}/yr</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_ef2:
                    promo_flag = engineered_df['Promotion_Delay_Flag'].iloc[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Promotion Delay</div>
                        <div class="metric-value" style="font-size:1.3rem; color:{'#ef4444' if promo_flag==1 else '#22c55e'};">{'Stalled (>3y)' if promo_flag==1 else 'Normal'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_ef3:
                    eng_comp = engineered_df['Engagement_Composite'].iloc[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Engagement Score</div>
                        <div class="metric-value" style="font-size:1.3rem;">{eng_comp:.2f} / 4.0</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_ef4:
                    stress_flag = engineered_df['Workload_Stress_Flag'].iloc[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Workload Stress</div>
                        <div class="metric-value" style="font-size:1.3rem; color:{'#ef4444' if stress_flag==1 else '#22c55e'};">{'High Stress' if stress_flag==1 else 'Balanced'}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("💡 Tailored Retention & Management Actions")
            recommendations = []
            
            if overtime == 'Yes' and work_life_balance <= 2:
                recommendations.append("⏰ **Relieve Critical OverTime Stress**: Employee is working overtime while experiencing low work-life balance. Consider load balancing or hiring contract support.")
            if years_since_promo > 3:
                recommendations.append(f"📈 **Promotion & Career Pathing**: Employee has gone {years_since_promo} years without promotion. Schedule a career advancement roadmap review.")
            if job_satisfaction <= 2 or env_satisfaction <= 2:
                recommendations.append("💬 **1-on-1 Managerial Sync**: Satisfaction scores are in the bottom tier (<=2). Conduct a skip-level or empathetic 1-on-1 to identify friction points.")
            if monthly_income < 3500 and total_working_years > 3:
                recommendations.append("💵 **Market Compensation Review**: Monthly income is lower than the cohort average given the years of experience.")
            if stock_option == 0 and job_level >= 2:
                recommendations.append("🪙 **Equity Incentive**: Consider adding stock option grants (LTI) to align long-term incentives.")
            if distance > 15 and business_travel == 'Travel_Frequently':
                recommendations.append("🏠 **Hybrid / Remote Flexibility**: Long commute (>15 miles) combined with frequent travel dramatically accelerates burnout.")
                
            if not recommendations:
                recommendations.append("✨ **Keep Doing What You're Doing**: Profile demonstrates solid satisfaction, balanced workload, and competitive alignment. Continue regular quarterly development check-ins.")
                
            for rec in recommendations:
                st.markdown(f"""<div class="recommendation-pill">{rec}</div>""", unsafe_allow_html=True)
                
        except Exception as ex:
            st.error(f"Prediction failed: {ex}")

elif menu == "📂 Batch Employee Analysis":
    st.subheader("Batch Workforce Screening & CSV Upload")
    st.write("Upload a CSV with employee attributes to score turnover risk across entire teams or departments.")
    
    col_up, col_btn = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload Employee Records (CSV)", type=["csv"])
    with col_btn:
        st.write("")
        st.write("")
        if st.button("📥 Load Demo Dataset Sample", width='stretch'):
            if dataset_df is not None:
                st.session_state['batch_df'] = dataset_df.drop(columns=['Attrition'], errors='ignore').head(100)
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.session_state['batch_df'] = batch_df
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            
    active_batch = st.session_state.get('batch_df', None)
    
    if active_batch is not None:
        st.success(f"Loaded {len(active_batch)} employee records for analysis.")
        
        with st.spinner("Processing feature engineering and calculating XGBoost risk probabilities..."):
            eng_batch = engineer_features(active_batch)
            X_batch_trans = preprocessor.transform(eng_batch)
            preds = model.predict(X_batch_trans)
            probs = model.predict_proba(X_batch_trans)[:, 1]
            
            results_df = active_batch.copy()
            results_df['Attrition_Prediction'] = np.where(preds == 1, "High Risk (Leaves)", "Low Risk (Stays)")
            results_df['Turnover_Risk_%'] = np.round(probs * 100, 1)
            results_df['Risk_Level'] = pd.cut(
                results_df['Turnover_Risk_%'],
                bins=[-1, 30, 60, 100],
                labels=['Low (<30%)', 'Moderate (30-60%)', 'High (>60%)']
            )
            
        high_risk_count = (results_df['Risk_Level'] == 'High (>60%)').sum()
        med_risk_count = (results_df['Risk_Level'] == 'Moderate (30-60%)').sum()
        low_risk_count = (results_df['Risk_Level'] == 'Low (<30%)').sum()
        avg_risk = results_df['Turnover_Risk_%'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Evaluated</div>
                <div class="metric-value">{len(results_df)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">High Risk Count</div>
                <div class="metric-value" style="color:#ef4444;">{high_risk_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Moderate Risk Count</div>
                <div class="metric-value" style="color:#f59e0b;">{med_risk_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average Risk Score</div>
                <div class="metric-value">{avg_risk:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_pie = px.pie(
                results_df, 
                names='Risk_Level', 
                title='Workforce Turnover Risk Tier Breakdown',
                color='Risk_Level',
                color_discrete_map={'Low (<30%)': '#22c55e', 'Moderate (30-60%)': '#f59e0b', 'High (>60%)': '#ef4444'},
                hole=0.45
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'})
            st.plotly_chart(fig_pie, width='stretch')
            
        with col_c2:
            if 'Department' in results_df.columns:
                dept_risk = results_df.groupby('Department')['Turnover_Risk_%'].mean().reset_index()
                fig_bar = px.bar(
                    dept_risk, 
                    x='Department', 
                    y='Turnover_Risk_%', 
                    title='Average Turnover Risk by Department',
                    color='Turnover_Risk_%',
                    color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444']
                )
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'})
                st.plotly_chart(fig_bar, width='stretch')
                
        st.subheader("📋 Scored Employee Registry")
        filter_tier = st.multiselect("Filter by Risk Level", ['High (>60%)', 'Moderate (30-60%)', 'Low (<30%)'], default=['High (>60%)', 'Moderate (30-60%)', 'Low (<30%)'])
        display_df = results_df[results_df['Risk_Level'].isin(filter_tier)]
        
        st.dataframe(
            display_df[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'Turnover_Risk_%', 'Risk_Level', 'Attrition_Prediction']],
            width='stretch'
        )
        
        csv_download = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Scored Predictions CSV",
            data=csv_download,
            file_name="workforce_attrition_predictions.csv",
            mime="text/csv",
            type="primary"
        )

elif menu == "📊 Workforce EDA & Trends":
    st.subheader("Palo Alto Networks / IBM HR Dataset Exploration")
    
    if dataset_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        total_emp = len(dataset_df)
        attr_count = dataset_df['Attrition'].sum()
        attr_rate = (attr_count / total_emp) * 100
        avg_inc = dataset_df['MonthlyIncome'].mean()
        avg_tenure = dataset_df['YearsAtCompany'].mean()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{total_emp:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Baseline Attrition Rate</div>
                <div class="metric-value" style="color:#ef4444;">{attr_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Mean Monthly Income</div>
                <div class="metric-value">${avg_inc:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Mean Company Tenure</div>
                <div class="metric-value">{avg_tenure:.1f} yrs</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ot_attr = dataset_df.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
            ot_attr['Attrition_Label'] = ot_attr['Attrition'].map({0: 'Stayed', 1: 'Left'})
            fig_ot = px.bar(
                ot_attr, 
                x='OverTime', 
                y='Count', 
                color='Attrition_Label', 
                barmode='group',
                title='Impact of Overtime on Employee Attrition',
                color_discrete_map={'Stayed': '#38bdf8', 'Left': '#ef4444'}
            )
            fig_ot.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'})
            st.plotly_chart(fig_ot, width='stretch')
            
        with col_g2:
            fig_box = px.box(
                dataset_df, 
                x='Department', 
                y='MonthlyIncome', 
                color=dataset_df['Attrition'].map({0: 'Stayed', 1: 'Left'}),
                title='Monthly Income Distribution by Department & Attrition',
                color_discrete_map={'Stayed': '#38bdf8', 'Left': '#ef4444'}
            )
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'})
            st.plotly_chart(fig_box, width='stretch')
            
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            role_attr = dataset_df.groupby('JobRole')['Attrition'].mean().reset_index()
            role_attr['Attrition_Rate'] = role_attr['Attrition'] * 100
            role_attr = role_attr.sort_values('Attrition_Rate', ascending=True)
            fig_role = px.bar(
                role_attr, 
                x='Attrition_Rate', 
                y='JobRole', 
                orientation='h',
                title='Attrition Rate by Job Role (%)',
                color='Attrition_Rate',
                color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444']
            )
            fig_role.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'})
            st.plotly_chart(fig_role, width='stretch')
            
        with col_g4:
            wlb_sat = dataset_df.groupby(['WorkLifeBalance', 'JobSatisfaction'])['Attrition'].mean().reset_index()
            wlb_sat['Attrition_Rate'] = (wlb_sat['Attrition'] * 100).round(1)
            pivot_table = wlb_sat.pivot(index='WorkLifeBalance', columns='JobSatisfaction', values='Attrition_Rate')
            fig_heat = px.imshow(
                pivot_table, 
                text_auto=True, 
                title='Attrition Rate % (Work-Life Balance vs Job Satisfaction)',
                labels=dict(x="Job Satisfaction (1-4)", y="Work-Life Balance (1-4)", color="Attrition %"),
                color_continuous_scale='Reds'
            )
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'})
            st.plotly_chart(fig_heat, width='stretch')
    else:
        st.warning("Palo Alto Networks.csv dataset file not found in directory.")

elif menu == "🧠 Model Architecture & Importance":
    st.subheader("XGBoost Model Architecture & Top Predictors")
    
    st.markdown("""
    The predictive engine leverages **XGBoost (Extreme Gradient Boosting)** coupled with an automated scikit-learn preprocessing pipeline.
    
    ### Pipeline Workflow:
    1. **Data Ingestion**: 30 raw demographic, role, compensation, and satisfaction attributes.
    2. **HR Feature Engineering**:
       - `Income_to_Experience = MonthlyIncome / (TotalWorkingYears + 1)`
       - `Promotion_Delay_Flag = I(YearsSinceLastPromotion > 3)`
       - `Engagement_Composite = mean(Involvement, Satisfaction, Environment, Relationship)`
       - `Workload_Stress_Flag = I(OverTime == 'Yes' ∧ WorkLifeBalance ≤ 2)`
    3. **Transformation**:
       - **Numeric Features (27)**: Scaled via `StandardScaler` (Zero Mean, Unit Variance).
       - **Categorical Features (7)**: Encoded via `OneHotEncoder(handle_unknown='ignore')`.
       - **Output Dimensionality**: 55 one-hot & scaled feature columns.
    4. **Inference**: Gradient boosted trees trained with logloss objective (`n_estimators=200`, `learning_rate=0.05`).
    """)
    
    st.markdown("---")
    
    if hasattr(model, 'feature_importances_'):
        try:
            ohe = preprocessor.named_transformers_['cat']
            cat_cols = preprocessor.transformers_[1][2]
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            num_cols = preprocessor.transformers_[0][2]
            all_feature_names = list(num_cols) + list(cat_feature_names)
            
            importances = model.feature_importances_
            
            if len(all_feature_names) == len(importances):
                feat_df = pd.DataFrame({
                    'Feature': all_feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(20)
                
                fig_imp = px.bar(
                    feat_df.sort_values('Importance', ascending=True),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 20 XGBoost Feature Importances',
                    color='Importance',
                    color_continuous_scale=['#38bdf8', '#818cf8', '#c084fc']
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': 'Plus Jakarta Sans', 'color': '#f8fafc'},
                    height=600
                )
                st.plotly_chart(fig_imp, width='stretch')
        except Exception as e:
            st.info(f"Could not compute individual feature names: {e}")
            
    st.markdown("---")
    st.markdown("Developed with ❤️ using **Streamlit**, **Scikit-Learn**, and **XGBoost**.")
