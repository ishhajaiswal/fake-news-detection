# import streamlit as st
# import pickle
# import pandas as pd
# import re
# import string
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load trained models
# with open("vectorizer.pkl", "rb") as v:
#     vectorizer = pickle.load(v)
# with open("logistic_model.pkl", "rb") as m:
#     model = pickle.load(m)

# # Function to clean text
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\b\d{4,}\b', '', text)  # Remove long numbers (years, large figures)
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
#     return text


# # Streamlit UI
# st.title("Fake News Detector")
# st.write("Enter a news article to check if it's real or fake.")

# user_input = st.text_area("News Article")
# if st.button("Check News"):
#     if user_input.strip():
#         cleaned_text = clean_text(user_input)
#         text_vectorized = vectorizer.transform([cleaned_text])
#         prediction = model.predict(text_vectorized)[0]
#         result = "Real News ✅" if prediction == 1 else "Fake News ❌"
#         st.subheader(result)
#     else:
#         st.warning("Please enter a news article.")

import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained models
with open("vectorizer.pkl", "rb") as v:
    vectorizer = pickle.load(v)
with open("logistic_model.pkl", "rb") as m:
    model = pickle.load(m)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a news article to check if it's real or fake.")

user_input = st.text_area("News Article")
if st.button("Check News"):
    if user_input.strip():
        cleaned_text = clean_text(user_input)
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        st.subheader(result)
    else:
        st.warning("Please enter a news article.")
