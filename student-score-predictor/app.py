import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Student Exam Score Predictor", page_icon="🎓", layout="wide")

# 1. Generate & Cache Realistic Student Dataset
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 300
    study_hours = np.round(np.random.uniform(1.0, 10.0, n_samples), 1)
    attendance = np.random.randint(60, 101, n_samples)
    previous_score = np.random.randint(40, 98, n_samples)
    sleep_hours = np.round(np.random.uniform(4.5, 9.0, n_samples), 1)

    exam_score = (
        10
        + (study_hours * 4.2)
        + (attendance * 0.25)
        + (previous_score * 0.30)
        + (sleep_hours * 1.1)
        + np.random.normal(0, 3.5, n_samples)
    )
    exam_score = np.clip(np.round(exam_score, 1), 0, 100)

    df = pd.DataFrame({
        "Study_Hours_Per_Day": study_hours,
        "Attendance_Percentage": attendance,
        "Previous_Exam_Score": previous_score,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Final_Exam_Score": exam_score
    })
    return df

df = load_data()

# 2. Train Machine Learning Model (Linear Regression)
X = df[["Study_Hours_Per_Day", "Attendance_Percentage", "Previous_Exam_Score", "Sleep_Hours_Per_Night"]]
y = df["Final_Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 3. Dashboard UI
st.title("🎓 Student Exam Score Predictor (Machine Learning)")
st.markdown("Predict a student's final exam score based on study habits, attendance, sleep, and past performance using **Multiple Linear Regression**.")

# Sidebar Inputs for Live Prediction
st.sidebar.header("📝 Enter Student Details")
in_study = st.sidebar.slider("Daily Study Hours", 1.0, 10.0, 5.5, 0.5)
in_attendance = st.sidebar.slider("Class Attendance (%)", 60, 100, 85, 1)
in_prev_score = st.sidebar.slider("Previous Exam Score", 40, 100, 72, 1)
in_sleep = st.sidebar.slider("Sleep Hours per Night", 4.0, 9.0, 7.0, 0.5)

input_data = pd.DataFrame([[in_study, in_attendance, in_prev_score, in_sleep]], columns=X.columns)
predicted_score = float(np.clip(model.predict(input_data)[0], 0, 100))

# KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Predicted Final Score", f"{predicted_score:.1f} / 100")
col2.metric("📈 Model Accuracy (R² Score)", f"{r2 * 100:.1f}%")
col3.metric("📉 Mean Absolute Error", f"± {mae:.2f} pts")
col4.metric("📚 Training Dataset Size", f"{len(df)} Students")

st.divider()

# Charts Row
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Study Hours vs. Final Exam Score")
    fig1 = px.scatter(
        df,
        x="Study_Hours_Per_Day",
        y="Final_Exam_Score",
        color="Attendance_Percentage",
        trendline="ols",
        labels={"Study_Hours_Per_Day": "Daily Study Hours", "Final_Exam_Score": "Final Exam Score"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.subheader("Actual vs. Predicted Scores (Test Set)")
    results_df = pd.DataFrame({"Actual Score": y_test, "Predicted Score": np.round(y_pred, 1)})
    fig2 = px.scatter(
        results_df,
        x="Actual Score",
        y="Predicted Score",
        labels={"Actual Score": "Actual Exam Score", "Predicted Score": "ML Predicted Score"}
    )
    fig2.add_shape(type="line", x0=40, y0=40, x1=100, y1=100, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)

# Feature Importance & Dataset Preview
st.subheader("🔍 What Impacts Exam Scores the Most? (Model Weights)")
coef_df = pd.DataFrame({
    "Feature": [
        "Daily Study Hours (+1 hr)",
        "Sleep Hours (+1 hr)",
        "Previous Score (+1 pt)",
        "Attendance (+1%)"
    ],
    "Score Impact (Points)": [
        round(float(model.coef_[0]), 2),
        round(float(model.coef_[3]), 2),
        round(float(model.coef_[2]), 2),
        round(float(model.coef_[1]), 2)
    ]
})
st.dataframe(coef_df, use_container_width=True, hide_index=True)

with st.expander("📄 View Raw Student Dataset & Download CSV"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Dataset as CSV", csv, "student_exam_data.csv", "text/csv")
