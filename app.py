import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text preprocessing
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Streamlit page configuration
st.set_page_config(page_title="Smart Spam Classifier", page_icon="📬", layout="wide")

# ------------------- APP TITLE -------------------
st.title("📬 Smart Spam Classifier")
st.markdown("### 🤖 ML + NLP Powered Email Classifier")
st.info("Detect spam emails in real-time with an interactive ML model!")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("📩 Try Examples")
    if st.button("🎁 Show SPAM Example"):
        st.session_state['email'] = "Congratulations! You've been selected for a FREE vacation! Click here."
    if st.button("💼 Show Not Spam"):
        st.session_state['email'] = "Hi John, please review the attached project file and revert by EOD."
    st.markdown("---")
    st.caption("📌 This application is developed for educational and demonstration purposes.")

# ------------------- MAIN AREA -------------------
st.subheader("✉️ Paste Email Content Below:")
user_input = st.text_area("", value=st.session_state.get("email", ""), height=150)

if st.button("🚀 Classify Message"):
    if not user_input.strip():
        st.warning("🚨 Please enter an email message first!")
    else:
        cleaned = clean_text(user_input)
        X = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(X)[0]
        probas = model.predict_proba(X)[0]

        label_text = "🚫 SPAM" if prediction == 1 else "✅ NOT SPAM"
        confidence = probas[prediction] * 100

        # --- OUTPUT METRICS ---
        st.subheader("📌 Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classification", label_text)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        # --- PROBABILITY CHART ---
        st.subheader("📊 Prediction Breakdown")
        fig, ax = plt.subplots()
        ax.bar(["Not Spam", "Spam"], probas * 100, color=["green", "red"])
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Classification Confidence")
        st.pyplot(fig)

        # --- CLEANED TEXT ---
        st.subheader("🧹 Preprocessed Message")
        st.code(cleaned)

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("Made with 🧠 NLTK, Scikit-learn & Streamlit | For academic and research purposes only.\n\n👨‍💻 Developed by Devpreet Singh")
