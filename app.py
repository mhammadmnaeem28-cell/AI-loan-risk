import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from datetime import datetime

# Dashboard Configuration
st.set_page_config(page_title="Loan Risk System", layout="wide")
st.title("🏦 Intelligent Loan Risk Assessment System")
st.markdown("---")

# Load Model and Data
# Saved model aur original data load kar rahe hain
data_pack = pickle.load(open('loan_model.pkl', 'rb'))
model, accuracy = data_pack['model'], data_pack['acc']
raw_df = pd.read_csv('loan_data.csv.csv')

# --- Indicators Section ---
# Dashboard ke shuru mein ahem maloomat
c1, c2, c3 = st.columns(3)
c1.metric("System Accuracy", f"{accuracy*100:.1f}%")
c2.metric("Project Status", "Online")
c3.metric("Dataset Size", f"{len(raw_df)} Rows")

st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("📝 Registration Details")
cust_name = st.sidebar.text_input("Full Name", "Muhammad Hammad")
cust_id = st.sidebar.text_input("Customer ID", "ID-2024-045")
city = st.sidebar.text_input("City", "Karachi")
report_date = datetime.now().strftime("%d-%m-%Y")

st.sidebar.header("💰 Financial Inputs")
age = st.sidebar.number_input("Age", 18, 100, 28)
income = st.sidebar.number_input("Annual Income ($)", 1000, 1000000, 65000)
emp_len = st.sidebar.slider("Work Experience (Years)", 0, 40, 6)
loan_amt = st.sidebar.number_input("Loan Amount Requested ($)", 500, 50000, 15000)
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 11.5)
home = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
intent = st.sidebar.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"])
grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D"])
history = st.sidebar.slider("Credit History (Years)", 1, 30, 5)
default = st.sidebar.radio("Previous Default?", ["N", "Y"])

# --- Prediction & Bill Display ---
if st.sidebar.button("Generate Assessment Report"):
    # Input data frame banana model ke liye
    input_df = pd.DataFrame([[age, income, home, emp_len, intent, grade, loan_amt, int_rate, loan_amt/income, default, history]], 
                            columns=model.feature_names_in_)
    
    # Risk calculation
    # Model se default hone ka chance (probability) le rahe hain
    prob = model.predict_proba(input_df)[0][1]
    
    # Risk Levels logic
    if prob < 0.30: risk, status, color = "LOW", "Highly Recommended", "#12be3a"
    elif prob < 0.65: risk, status, color = "MEDIUM", "Conditional Approval", "#fd7e14"
    else: risk, status, color = "HIGH", "Rejected / High Risk", "#dc3545"

    # --- SIMPLE RESULT ---
    st.subheader("📋 Loan Assessment Result")
    if risk == "LOW":
        st.success(f"✅ Loan APPROVED — LOW RISK  |  {status}  |  Risk Score: {prob*100:.1f}%")
    elif risk == "MEDIUM":
        st.warning(f"⚠️ Loan CONDITIONAL — MEDIUM RISK  |  {status}  |  Risk Score: {prob*100:.1f}%")
    else:
        st.error(f"❌ Loan REJECTED — HIGH RISK  |  {status}  |  Risk Score: {prob*100:.1f}%")

st.divider()

# --- Visualizations ---
# Graphs aur charts display karna
st.subheader("📊 System Insights (Real Data Analysis)")
col_a, col_b = st.columns(2)

with col_a:
    fig1 = px.histogram(raw_df, x="loan_intent", color="loan_status", barmode="group", title="Loan Status by Intent")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    # Scatter plot ko define aur display dono karna hai
    fig2 = px.scatter(raw_df.sample(800, random_state=42), 
                      x="person_age", 
                      y="person_income", 
                      color="loan_status", 
                      title="Age vs Income Risk Scatter")
    st.plotly_chart(fig2, use_container_width=True)


# =====================================================================
# NAYE GRAPHS - 3 aur graphs add kar rahe hain
# =====================================================================

st.divider()
st.subheader("📈 Additional Data Insights")

# --- Graph 3: Pie Chart - Home Ownership Distribution ---
# Yeh graph dikhata hai kitne log RENT, OWN, ya MORTGAGE mein rehte hain
col_c, col_d = st.columns(2)

with col_c:
    fig3 = px.pie(
        raw_df,
        names="person_home_ownership",
        title="Home Ownership Distribution",
        hole=0.3  # donut style pie chart
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- Graph 4: Bar Chart - Average Loan Amount by Loan Grade ---
# Har grade (A, B, C, D) ke liye average loan amount dikhata hai
with col_d:
    avg_loan_by_grade = raw_df.groupby("loan_grade")["loan_amnt"].mean().reset_index()
    avg_loan_by_grade.columns = ["loan_grade", "avg_loan_amount"]

    fig4 = px.bar(
        avg_loan_by_grade,
        x="loan_grade",
        y="avg_loan_amount",
        color="loan_grade",
        title="Average Loan Amount by Grade",
        text_auto=".2s"  # bar ke upar value dikhao
    )
    st.plotly_chart(fig4, use_container_width=True)

# --- Graph 5: MERGED GRAPH - Bar + Line dono ek hi graph mein ---
# Same X axis (Loan Grade) aur same Y axis (Average Loan Amount)
# Bar graph = average loan amount per grade
# Line graph = average interest rate per grade (trend line)

st.subheader("🔀 Merged Graph: Loan Grade → Avg Loan Amount (Bar) + Avg Interest Rate (Line)")

import plotly.graph_objects as go

# Loan grade ke hisaab se average loan amount aur average interest rate nikalna
grade_data = raw_df.groupby("loan_grade").agg(
    avg_loan=("loan_amnt", "mean"),
    avg_rate=("loan_int_rate", "mean")
).reset_index()

fig5 = go.Figure()

# Bar graph - Average Loan Amount
fig5.add_trace(go.Bar(
    x=grade_data["loan_grade"],
    y=grade_data["avg_loan"],
    name="Avg Loan Amount ($)",
    marker_color="#636EFA"
))

# Line graph - Average Interest Rate (same X, same Y axis)
fig5.add_trace(go.Scatter(
    x=grade_data["loan_grade"],
    y=grade_data["avg_rate"],
    name="Avg Interest Rate (%)",
    mode="lines+markers",
    line=dict(color="#EF553B", width=3),
    marker=dict(size=8)
))

# Layout - ek hi X aur Y axis
fig5.update_layout(
    title="Loan Grade: Avg Loan Amount (Bar) & Avg Interest Rate (Line) — Same Axis",
    xaxis_title="Loan Grade",
    yaxis_title="Value",
    legend=dict(orientation="h", y=1.1),
    height=450
)

st.plotly_chart(fig5, use_container_width=True)