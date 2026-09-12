import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Risk Intelligence",
    page_icon="💼",
    layout="wide"
)

# -------------------------------------------------------------
# Load Data & Construct Preprocessor In-Memory (Zero Pickle Conflicts)
# -------------------------------------------------------------
@st.cache_resource
def load_model_and_pipeline():
    # 1. Load the trained XGBoost model
    model = joblib.load('best_model.pkl')
    
    # 2. Load and engineer the dataset
    df = pd.read_csv('Palo Alto Networks.csv')
    df.columns = df.columns.str.strip()
    
    # Feature Engineering (must match training exactly)
    df['Income_to_Experience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['Promotion_Delay_Flag'] = (df['YearsSinceLastPromotion'] >= 4).astype(int)
    
    sat_cols = [c for c in ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance'] if c in df.columns]
    df['Engagement_Composite'] = df[sat_cols].mean(axis=1)
    
    if 'OverTime' in df.columns and 'JobInvolvement' in df.columns:
        df['Workload_Stress_Flag'] = ((df['OverTime'].astype(str).str.strip().str.capitalize() == 'Yes') & (df['JobInvolvement'] <= 2)).astype(int)
    
    # Remove unused columns
    drop_cols = ['EmployeeCount', 'StandardHours', 'Over18']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    if 'EmployeeId' not in df.columns and 'EmployeeNumber' not in df.columns:
        df['EmployeeID'] = [f"EMP-{i+1001}" for i in range(len(df))]
    elif 'EmployeeNumber' in df.columns:
        df['EmployeeID'] = "EMP-" + df['EmployeeNumber'].astype(str)

    # 3. Fit Preprocessor dynamically with the current environment's scikit-learn
    feature_cols = [c for c in df.columns if c not in ['Attrition', 'EmployeeID']]
    cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    preprocessor.fit(df[feature_cols])

    # 4. Predict probabilities across all records
    X_proc = preprocessor.transform(df[feature_cols])
    probs = model.predict_proba(X_proc)[:, 1]
    df['Attrition_Probability'] = np.round(probs, 4)

    def assign_tier(p):
        if p < 0.30:
            return 'Low Risk'
        elif p <= 0.60:
            return 'Medium Risk'
        else:
            return 'High Risk'

    df['Risk_Category'] = df['Attrition_Probability'].apply(assign_tier)
    
    return model, preprocessor, df, feature_cols

try:
    model, preprocessor, df, feature_cols = load_model_and_pipeline()
except Exception as e:
    st.error(f"Error initializing intelligence engine: {e}")
    st.stop()

# -------------------------------------------------------------
# Sidebar: Global Filters
# -------------------------------------------------------------
st.sidebar.header("Workforce Filters")

dept_list = ['All'] + sorted(df['Department'].dropna().unique().tolist())
selected_dept = st.sidebar.selectbox("Select Department", dept_list)

filtered_df = df.copy()
if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]

roles_list = ['All'] + sorted(filtered_df['JobRole'].dropna().unique().tolist())
selected_role = st.sidebar.selectbox("Select Job Role", roles_list)

if selected_role != 'All':
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]

risk_cutoff = st.sidebar.slider("High Risk Threshold Cutoff", min_value=0.50, max_value=0.90, value=0.60, step=0.05)

# -------------------------------------------------------------
# Main Header & KPIs
# -------------------------------------------------------------
st.title("🛡️ Predictive Workforce Intelligence Dashboard")
st.markdown("Early-warning decision engine to identify flight risk, diagnose turnover drivers, and simulate retention interventions.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_emp = len(filtered_df)
high_risk_count = (filtered_df['Attrition_Probability'] >= risk_cutoff).sum()
med_risk_count = ((filtered_df['Attrition_Probability'] >= 0.30) & (filtered_df['Attrition_Probability'] < risk_cutoff)).sum()
avg_prob = filtered_df['Attrition_Probability'].mean() if total_emp > 0 else 0

kpi1.metric("Filtered Headcount", f"{total_emp:,}")
kpi2.metric("High-Risk Count", f"{high_risk_count:,}", delta=f"{round((high_risk_count/total_emp)*100, 1) if total_emp else 0}% pool", delta_color="inverse")
kpi3.metric("Medium-Risk Count", f"{med_risk_count:,}")
kpi4.metric("Avg Flight Probability", f"{avg_prob:.1%}")

st.divider()

# -------------------------------------------------------------
# Navigation Tabs
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Macro Risk Analytics", 
    "👤 Individual Employee Profile", 
    "🧪 What-If Scenario Simulator"
])

# -------------------------------------------------------------
# TAB 1: Macro Analytics
# -------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Workforce Risk Distribution")
        risk_counts = filtered_df['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Tier', 'Count']
        
        color_map = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
        fig_donut = px.pie(
            risk_counts, 
            names='Risk Tier', 
            values='Count', 
            hole=0.55,
            color='Risk Tier',
            color_discrete_map=color_map
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Risk Exposure by Department")
        dept_risk = df.groupby(['Department', 'Risk_Category']).size().reset_index(name='Employees')
        fig_bar = px.bar(
            dept_risk, 
            x='Department', 
            y='Employees', 
            color='Risk_Category', 
            barmode='stack',
            color_discrete_map=color_map
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("High Flight-Risk Employees (Action List)")
    top_risks = filtered_df.sort_values(by='Attrition_Probability', ascending=False)[
        ['EmployeeID', 'JobRole', 'Department', 'MonthlyIncome', 'OverTime', 'YearsAtCompany', 'Attrition_Probability', 'Risk_Category']
    ].head(10)
    st.dataframe(top_risks.style.format({'Attrition_Probability': '{:.1%}', 'MonthlyIncome': '${:,.0f}'}), use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Individual Profile
# -------------------------------------------------------------
with tab2:
    st.subheader("Individual Diagnostic Dossier")
    
    selected_emp_id = st.selectbox("Select Employee ID", options=filtered_df['EmployeeID'].tolist())
    emp_record = filtered_df[filtered_df['EmployeeID'] == selected_emp_id].iloc[0]
    
    p_col1, p_col2 = st.columns([1, 1])
    
    with p_col1:
        prob_val = emp_record['Attrition_Probability']
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_val * 100,
            title={'text': f"Attrition Risk: {emp_record['Risk_Category']}"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#111827"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},
                    {'range': [30, 60], 'color': "#f39c12"},
                    {'range': [60, 100], 'color': "#e74c3c"}
                ]
            }
        ))
        gauge_fig.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with p_col2:
        st.markdown(f"**Role:** {emp_record['JobRole']}")
        st.markdown(f"**Department:** {emp_record['Department']}")
        st.markdown(f"**Monthly Income:** ${emp_record['MonthlyIncome']:,}")
        st.markdown(f"**Tenure at Company:** {emp_record['YearsAtCompany']} years")
        st.markdown(f"**OverTime Active:** {'Yes' if emp_record['OverTime'] == 'Yes' else 'No'}")
        st.markdown(f"**Years Since Last Promotion:** {emp_record['YearsSinceLastPromotion']} years")

    st.markdown("#### Primary Attrition Signals")
    reasons = []
    if emp_record['OverTime'] == 'Yes':
        reasons.append("⚠️ **Excessive OverTime**: Frequent overtime significantly accelerates turnover intent.")
    if emp_record['YearsSinceLastPromotion'] >= 4:
        reasons.append(f"⚠️ **Promotion Stagnation**: Has not received a promotion in {emp_record['YearsSinceLastPromotion']} years.")
    if emp_record['JobSatisfaction'] <= 2:
        reasons.append("⚠️ **Low Job Satisfaction**: Sub-optimal contentment registered in review.")
    if emp_record['WorkLifeBalance'] <= 2:
        reasons.append("⚠️ **Strained Work-Life Balance**: Elevates acute burnout probability.")
    if emp_record['Income_to_Experience'] < 300:
        reasons.append("⚠️ **Compensation Disparity**: Monthly salary is lower relative to total career experience.")

    if reasons:
        for r in reasons:
            st.markdown(r)
    else:
        st.success("No critical attrition flags identified. Compensation, role progression, and satisfaction levels align with retention standards.")

# -------------------------------------------------------------
# TAB 3: What-If Simulator
# -------------------------------------------------------------
with tab3:
    st.subheader("Intervention Sandbox")
    st.caption("Simulate proactive retention counter-measures and immediately observe risk adjustments.")

    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        current_salary = int(emp_record['MonthlyIncome'])
        new_salary = st.slider("Adjust Monthly Salary ($)", min_value=1000, max_value=25000, value=current_salary, step=500)
        
        current_ot = emp_record['OverTime']
        new_ot = st.selectbox("Overtime Status", options=['No', 'Yes'], index=0 if current_ot == 'No' else 1)

    with sim_col2:
        current_wlb = int(emp_record['WorkLifeBalance'])
        new_wlb = st.slider("Adjust Work-Life Balance (1-4)", min_value=1, max_value=4, value=current_wlb)
        
        current_prom = int(emp_record['YearsSinceLastPromotion'])
        new_prom = st.slider("Years Since Last Promotion (0 = Promoted Today)", min_value=0, max_value=15, value=current_prom)

    sim_data = emp_record[feature_cols].copy().to_frame().T
    sim_data['MonthlyIncome'] = new_salary
    sim_data['OverTime'] = new_ot
    sim_data['WorkLifeBalance'] = new_wlb
    sim_data['YearsSinceLastPromotion'] = new_prom
    
    sim_data['Income_to_Experience'] = new_salary / (int(sim_data['TotalWorkingYears'].iloc[0]) + 1)
    sim_data['Promotion_Delay_Flag'] = int(new_prom >= 4)
    sim_data['Workload_Stress_Flag'] = int((new_ot == 'Yes') and (int(sim_data['JobInvolvement'].iloc[0]) <= 2))
    
    sim_proc = preprocessor.transform(sim_data)
    new_prob = float(model.predict_proba(sim_proc)[0, 1])
    delta_prob = (new_prob - emp_record['Attrition_Probability']) * 100

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Baseline Risk", f"{emp_record['Attrition_Probability']:.1%}")
    res_col2.metric("Simulated Post-Intervention Risk", f"{new_prob:.1%}", delta=f"{delta_prob:+.1f}%", delta_color="inverse")
