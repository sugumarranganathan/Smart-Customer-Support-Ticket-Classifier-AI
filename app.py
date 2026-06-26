import streamlit as st
import joblib

# -----------------------------
# Load Saved Model & Vectorizer
# -----------------------------
model = joblib.load("customer_support_model.pkl")
vectorizer = joblib.load("hashing_vectorizer.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎫 Customer Support Ticket Classification")
st.write(
    "Predict the issue category of a customer support ticket using "
    "**Hashing Vectorizer** and **Multinomial Naive Bayes**."
)

# -----------------------------
# Text Input
# -----------------------------
ticket = st.text_area(
    "Enter Customer Support Ticket",
    height=180,
    placeholder="Example: I forgot my account password and cannot login."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Category"):

    if ticket.strip() == "":
        st.warning("Please enter a customer support ticket.")
    else:
        ticket_vector = vectorizer.transform([ticket])
        prediction = model.predict(ticket_vector)[0]

        st.success(f"Predicted Category: **{prediction}**")

# -----------------------------
# Sample Examples
# -----------------------------
st.markdown("---")
st.subheader("Sample Tickets")

st.code("I forgot my account password and cannot login.")
st.code("Payment was deducted twice from my account.")
st.code("The application crashes whenever I open it.")
st.code("I think someone used my card without permission.")
st.code("I have a question about your subscription plans.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Developed using Python, Streamlit, Hashing Vectorizer, and Multinomial Naive Bayes."
)
