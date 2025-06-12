import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

st.set_page_config(page_title="🌍 Data Science Salary Predictor", layout="wide")
st.title("💼 Data Science Salary Predictor")

@st.cache_data
def load_data():
    return pd.read_csv("data_science_salaries.csv")

df = load_data()

# Feature engineering
df["is_remote"] = df["work_models"].apply(lambda x: 1 if x == "Remote" else 0)

def bin_salary(sal):
    if sal < 50000:
        return "Low"
    elif sal < 100000:
        return "Medium"
    else:
        return "High"

df["salary_bin"] = df["salary_in_usd"].apply(bin_salary)

# Sidebar filters
st.sidebar.header("🔧 Filter Options")
selected_job = st.sidebar.selectbox("Job Title", sorted(df['job_title'].unique()))
selected_exp = st.sidebar.selectbox("Experience Level", sorted(df['experience_level'].unique()))
selected_emp = st.sidebar.selectbox("Employment Type", sorted(df['employment_type'].unique()))
selected_model = st.sidebar.selectbox("Work Model", sorted(df['work_models'].unique()))
selected_loc = st.sidebar.selectbox("Employee Residence", sorted(df['employee_residence'].unique()))
selected_cloc = st.sidebar.selectbox("Company Location", sorted(df['company_location'].unique()))
selected_size = st.sidebar.selectbox("Company Size", sorted(df['company_size'].unique()))
selected_currency = st.sidebar.selectbox("Salary Currency", sorted(df['salary_currency'].unique()))

# Filtered preview
filtered_df = df[
    (df['job_title'] == selected_job) &
    (df['experience_level'] == selected_exp) &
    (df['employment_type'] == selected_emp) &
    (df['work_models'] == selected_model) &
    (df['employee_residence'] == selected_loc) &
    (df['company_location'] == selected_cloc) &
    (df['company_size'] == selected_size) &
    (df['salary_currency'] == selected_currency)
]

st.subheader("🔍 Filtered Data")
st.dataframe(filtered_df.head())

# Graph 1: Salary distribution
st.subheader("📊 Salary Distribution (USD)")
fig1, ax1 = plt.subplots()
sns.histplot(df["salary_in_usd"], bins=40, kde=True, ax=ax1)
st.pyplot(fig1)

# Graph 2: Salary bin distribution
st.subheader("📈 Salary Range Distribution")
fig2, ax2 = plt.subplots()
df["salary_bin"].value_counts().plot(kind="bar", color=["#2ecc71", "#f1c40f", "#e74c3c"], ax=ax2)
plt.xlabel("Salary Range")
plt.ylabel("Count")
st.pyplot(fig2)

# Model training
df_model = df.drop(columns=["salary", "salary_bin", "salary_currency"])
df_encoded = pd.get_dummies(df_model, drop_first=True)

X = df_encoded.drop("salary_in_usd", axis=1)
y = df_encoded["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Evaluation
preds_usd = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds_usd))
r2 = r2_score(y_test, preds_usd)

def pred_bin(val):
    if val < 50000:
        return "Low"
    elif val < 100000:
        return "Medium"
    else:
        return "High"

y_test_bins = y_test.apply(bin_salary)
pred_bins = pd.Series(preds_usd).apply(pred_bin)
bin_accuracy = accuracy_score(y_test_bins, pred_bins)

# Metric display
st.subheader("📏 Model Performance")
st.markdown("""
<style>
.metric-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}
.metric-title {
    font-size: 20px;
    font-weight: bold;
}
.metric-value {
    font-size: 28px;
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-box">
    <div class="metric-title">✅ Classification Accuracy</div>
    <div class="metric-value">{bin_accuracy*100:.2f}%</div>
</div>
<div class="metric-box">
    <div class="metric-title">📈 R² Score</div>
    <div class="metric-value">{r2:.2f}</div>
</div>
<div class="metric-box">
    <div class="metric-title">📉 RMSE</div>
    <div class="metric-value">{rmse:,.2f} USD</div>
</div>
""", unsafe_allow_html=True)

# Confusion matrix
st.subheader("🧮 Confusion Matrix (Salary Range Prediction)")
cm = confusion_matrix(y_test_bins, pred_bins, labels=["Low", "Medium", "High"])
cm_df = pd.DataFrame(cm, index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"])

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax_cm)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

# Graph 3: Feature importance
st.subheader("🎯 Feature Importance")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig3, ax3 = plt.subplots()
importance.head(10).plot(kind='barh', ax=ax3)
plt.xlabel("Importance Score")
st.pyplot(fig3)

# Prediction
st.subheader("💰 Predict Salary in Selected Currency")

user_input = {
    'job_title': selected_job,
    'experience_level': selected_exp,
    'employment_type': selected_emp,
    'work_models': selected_model,
    'employee_residence': selected_loc,
    'company_location': selected_cloc,
    'company_size': selected_size,
    'work_year': df['work_year'].max(),
    'is_remote': 1 if selected_model == "Remote" else 0
}

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

if st.button("🎯 Predict Salary"):
    pred_usd = model.predict(input_encoded)[0]

    try:
        local_mean = df[df['salary_currency'] == selected_currency]['salary'].astype(float).mean()
        usd_mean = df[df['salary_currency'] == selected_currency]['salary_in_usd'].mean()
        conversion_rate = usd_mean / local_mean if local_mean else 1
    except:
        conversion_rate = 1

    pred_local = pred_usd / conversion_rate

    st.success(f"🌍 Predicted Salary in USD: ${pred_usd:,.2f}")
    st.success(f"💱 Estimated Salary in {selected_currency}: {pred_local:,.2f} {selected_currency}")
