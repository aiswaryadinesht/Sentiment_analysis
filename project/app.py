import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing function
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('/path/to/test.csv')
    return data

# Train model
@st.cache_resource
def train_model(data):
    data['cleaned_review'] = data['review'].apply(preprocess_text)

    # Split dataset
    X = data['cleaned_review']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization with TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy, y_test, y_pred

# Main app
data = load_data()
model, vectorizer, accuracy, y_test, y_pred = train_model(data)

st.title(" Sentiment Analysis App")

# st.write("### Dataset Sample")
# st.write(data.head())

st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report and confusion matrix
# st.write("### Classification Report")
# st.text(classification_report(y_test, y_pred))

# st.write("### Confusion Matrix")
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# st.pyplot()

# User input
st.write("### Enter a review to classify:")
user_input = st.text_area("Review:")
if st.button("Predict"):
    if user_input:
        cleaned_input = preprocess_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]
        st.write(f"**Predicted Sentiment:** {prediction.capitalize()}")
    else:
        st.write("Please enter a review.")
