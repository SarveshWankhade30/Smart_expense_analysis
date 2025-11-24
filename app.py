import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained ML model and vectorizer
model = joblib.load("expense_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Smart Expense Categorizer", layout="wide")
st.title("💰 Smart Expense Categorizer")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your expense CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Expense Data")
    st.dataframe(df)

    # --- Summary Metrics ---
    st.subheader("📈 Expense Summary")
    total = df["Amount"].sum()
    st.metric("Total Expense", f"₹{total}")

    # Category-wise pie chart
    cat_sum = df.groupby("Category")["Amount"].sum()
    fig1, ax1 = plt.subplots()
    cat_sum.plot(kind="pie", autopct="%1.1f%%", ax=ax1)
    st.pyplot(fig1)

    # Monthly trend
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%b')
    monthly_sum = df.groupby('Month')['Amount'].sum().reindex(
        ['Jan','Feb','Mar','Apr','May','Jun']
    )
    st.subheader("📅 Monthly Expense Trend")
    fig2, ax2 = plt.subplots()
    monthly_sum.plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_ylabel("Amount (₹)")
    st.pyplot(fig2)

# --- Add New Expense ---
st.subheader("➕ Add New Expense")
with st.form("add_expense"):
    date = st.date_input("Date")
    desc = st.text_input("Description")
    amt = st.number_input("Amount", min_value=1)
    submitted = st.form_submit_button("Predict Category")

    if submitted:
        # Predict category
        desc_vec = vectorizer.transform([desc])
        predicted_category = model.predict(desc_vec)[0]
        st.success(f"Predicted Category: {predicted_category}")


# a = [4, 5, 5, 9]
# cov = [i ** 2 for i in a]
# print(cov)