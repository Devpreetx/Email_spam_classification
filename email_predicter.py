import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk

nltk.download('stopwords')

# Load and clean data
df = pd.read_csv("spam_email_250.csv")
ps = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

df['cleaned_text'] = df['message'].apply(clean_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()

# Encoding labels
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
