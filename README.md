# 📧 Spam Email Classification Web App

This project is a Machine Learning-based web application developed using Streamlit. It classifies whether an email is Spam or Not Spam based on the content provided. With a clean and interactive UI, users can input email text and instantly see the prediction in real-time.

## 🔍 Key Points

Spam Detection with NLP
Predicts whether an email is Spam or Not Spam using Natural Language Processing techniques.

Interactive Web Interface with Streamlit
Provides a simple, user-friendly interface where users can paste or type email text and get predictions—no coding required.

Machine Learning Model Integration
Trained with models like Naive Bayes, SVM, and Random Forest. The best-performing model is deployed for high accuracy.

Real-Time Predictions
The app instantly processes text and returns results, ensuring fast and reliable spam detection.

Deployment Ready
Easily deployable via Streamlit Community Cloud, GitHub, or any Python-compatible cloud platform.

✅ Features Used

The following NLP and preprocessing techniques were used to train the spam classifier model:

Tokenization – Breaking email text into words.

Stopword Removal – Removing irrelevant/common words (e.g., the, is, at).

Stemming – Reducing words to their root form (e.g., running → run).

TF-IDF Vectorization – Converting text into numerical feature vectors.

These steps help the model identify patterns in text that differentiate Spam emails from Non-Spam emails.

✅ Model Training Overview

Naive Bayes Classifier

Support Vector Machine (SVM)

Random Forest Classifier

Logistic Regression

Gradient Boosting

The best-performing model (based on accuracy & F1 score) was saved as spam_model.pkl using Joblib for future use in prediction without retraining.

🚀 Streamlit Web App
You can run the Streamlit frontend using the command:

streamlit run app.py
