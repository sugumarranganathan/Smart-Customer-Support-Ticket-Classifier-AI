# Customer Support Ticket Classification Using Hashing Vectorizer and Multinomial Naive Bayes

https://colab.research.google.com/drive/1Yu-i9JQG7RI6SFDjEl3LM0oPZmAe942u#scrollTo=g5CzvJ_JDSNH

https://sugumarranganathan-smart-customer-support-ticket-3abd45d.hf.space/

# 🎫 Smart Customer Support Ticket Classifier AI

## 📌 Project Overview

Smart Customer Support Ticket Classifier AI is a Natural Language Processing (NLP) application that automatically classifies customer support tickets into predefined issue categories using **Hashing Vectorizer** and **Multinomial Naive Bayes**.

The application analyzes the text entered by a user and predicts the most appropriate support category such as **Account**, **Billing**, **Fraud**, **General Inquiry**, or **Technical**. This helps organizations route customer tickets to the correct support team quickly and efficiently.

---

# 🎯 Problem Statement

Organizations receive hundreds or even thousands of customer support tickets every day through emails, websites, chat applications, and mobile apps.

Manually reading every ticket and assigning it to the appropriate department is:

* Time-consuming
* Error-prone
* Expensive
* Difficult to scale

Incorrect ticket routing delays issue resolution and reduces customer satisfaction.

This project automates the ticket classification process using Machine Learning, enabling faster and more accurate routing of customer requests.

---

# 💡 Proposed Solution

This application uses:

* Hashing Vectorizer for converting customer ticket text into numerical feature vectors.
* Multinomial Naive Bayes for classifying tickets into predefined categories.

The prediction is completed within seconds, reducing manual effort and improving customer support efficiency.

---

# 🚀 Features

* Automatic Customer Support Ticket Classification
* NLP-Based Text Processing
* Hashing Vectorizer Feature Extraction
* Multinomial Naive Bayes Classification
* Instant Prediction
* Easy-to-use Gradio Interface
* Ready for Hugging Face Deployment

---

# 📂 Issue Categories

* Account
* Billing
* Fraud
* General Inquiry
* Technical

---

# 🛠 Technologies Used

* Python
* Gradio
* Pandas
* NumPy
* Scikit-learn
* Hashing Vectorizer
* Multinomial Naive Bayes
* Joblib

---

# 📊 Dataset Information

* Dataset: Customer Support Tickets
* Total Records: 13,837+
* Input Feature: **Ticket_Description**
* Target Variable: **Issue_Category**

---

# 🔄 Project Workflow

```
Customer Support Ticket
          │
          ▼
Text Preprocessing
          │
          ▼
Hashing Vectorizer
          │
          ▼
Multinomial Naive Bayes
          │
          ▼
Predicted Issue Category
```

---

# 📈 Model Performance

| Metric    | Result |
| --------- | ------ |
| Accuracy  | 99.96% |
| Precision | 1.00   |
| Recall    | 1.00   |
| F1-Score  | 1.00   |

The model achieved excellent performance on the test dataset with only one misclassification.

---

# 🎯 Applications

* Customer Support Centers
* Banking Services
* E-commerce Platforms
* Telecommunication Companies
* Insurance Services
* Healthcare Support Systems
* IT Help Desk
* SaaS Customer Support

---

# ✅ Benefits

* Reduces manual ticket classification
* Faster customer response
* Improves routing accuracy
* Saves operational cost
* Reduces human errors
* Increases support team productivity
* Handles large volumes of customer requests efficiently

---

# 📁 Project Structure

```
Smart-Customer-Support-Ticket-Classifier-AI/
│
├── app.py
├── customer_support_model.pkl
├── hashing_vectorizer.pkl
├── requirements.txt
├── README.md
└── customer_support_tickets.csv
```

---

# ▶️ Installation

```bash
git clone https://github.com/your-username/Smart-Customer-Support-Ticket-Classifier-AI.git

cd Smart-Customer-Support-Ticket-Classifier-AI

pip install -r requirements.txt

python app.py
```

---

# 📌 Sample Input

```
I forgot my account password and cannot login.
```

### Predicted Output

```
Account
```

---

# 👨‍💻 Prepared By

**Sugumar Ranganathan**

MBA | AI & Data Analytics Enthusiast

---

# ⭐ If you found this project useful, please consider giving it a Star on GitHub!
