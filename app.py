import gradio as gr
import joblib

# ---------------------------------
# Load Model and Vectorizer
# ---------------------------------
model = joblib.load("customer_support_model.pkl")
vectorizer = joblib.load("hashing_vectorizer.pkl")

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_ticket(ticket):

    if ticket.strip() == "":
        return "⚠ Please enter a customer support ticket."

    ticket_vector = vectorizer.transform([ticket])

    prediction = model.predict(ticket_vector)[0]

    return f"Predicted Category: {prediction}"


# ---------------------------------
# Gradio Interface
# ---------------------------------
demo = gr.Interface(
    fn=predict_ticket,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Example: I forgot my account password and cannot login.",
        label="Enter Customer Support Ticket"
    ),
    outputs=gr.Textbox(
        label="Prediction"
    ),
    title="🎫 Smart Customer Support Ticket Classifier AI",
    description="""
Automatically classify customer support tickets using
Hashing Vectorizer and Multinomial Naive Bayes.

Supported Categories:
• Account
• Billing
• Fraud
• General Inquiry
• Technical
""",
    examples=[
        ["I forgot my account password and cannot login."],
        ["Payment deducted twice from my bank account."],
        ["Someone used my credit card without permission."],
        ["The application crashes every time I open it."],
        ["I have a question about my subscription."]
    ],
    theme=gr.themes.Soft()
)

demo.launch()
