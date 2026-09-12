# Workforce Attrition Risk Predictor & Analytics

An end-to-end Machine Learning web application powered by **XGBoost**, **Scikit-Learn**, and **Streamlit** to analyze and predict employee turnover risk.

## 🚀 Key Features
- **Individual Risk Predictor**: Evaluates turnover probability (0–100%) with Plotly gauge visualizations, feature diagnostics, and customized HR retention recommendations.
- **Batch CSV Analysis**: Process and score entire employee rosters with risk distribution breakdowns and exportable reports.
- **Workforce EDA & Trends**: Interactive charts exploring key attrition drivers such as overtime, compensation, and satisfaction metrics.
- **Feature Importance & Model Architecture**: Inspect the top decision drivers powering the trained XGBoost classifier.

## 🛠️ Tech Stack
- **Python 3.12**
- **Streamlit**
- **XGBoost**
- **Scikit-Learn**
- **Pandas & NumPy**
- **Plotly**

## 📦 Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Anmol007-007/ML_risk_pred.git
cd ML_risk_pred
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application:**
```bash
streamlit run app.py
```
