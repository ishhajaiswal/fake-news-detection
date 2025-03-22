# import pandas as pd
# import numpy as np
# import re
# import string
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load datasets
# fake_df = pd.read_csv("Fake.csv")
# true_df = pd.read_csv("True.csv")

# # Add labels
# fake_df["label"] = 0  # Fake news
# true_df["label"] = 1  # Real news

# # Combine both datasets
# df = pd.concat([fake_df, true_df], ignore_index=True)

# # Shuffle the dataset
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Data Cleaning
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# df['text'] = df['text'].apply(clean_text)

# # Splitting Data
# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# # Text Vectorization
# # vectorizer = TfidfVectorizer(max_features=5000)
# vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams & bigrams

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Naïve Bayes Model
# nb_model = MultinomialNB()
# nb_model.fit(X_train_tfidf, y_train)
# y_pred_nb = nb_model.predict(X_test_tfidf)

# # Logistic Regression Model
# lr_model = LogisticRegression()
# lr_model.fit(X_train_tfidf, y_train)
# y_pred_lr = lr_model.predict(X_test_tfidf)

# # Evaluation
# def evaluate_model(y_test, y_pred, model_name):
#     print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
#     print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# evaluate_model(y_test, y_pred_nb, "Naïve Bayes")
# evaluate_model(y_test, y_pred_lr, "Logistic Regression")

# # Save TF-IDF vectorizer
# with open("vectorizer.pkl", "wb") as v:
#     pickle.dump(vectorizer, v)

# # Save trained Logistic Regression model
# with open("logistic_model.pkl", "wb") as m:
#     pickle.dump(lr_model, m)

# print("Model and vectorizer saved successfully!")

import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine datasets
data = pd.concat([true_df, fake_df]).reset_index(drop=True)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning
data["text"] = data["title"] + " " + data["text"]  # Combine title & content
data["text"] = data["text"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)  # Unigrams & bigrams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)

print("Naïve Bayes Accuracy:", round(accuracy_score(y_test, nb_preds), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_preds))
print("Classification Report:")
print(classification_report(y_test, nb_preds))

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)
logistic_preds = logistic_model.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", round(accuracy_score(y_test, logistic_preds), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_preds))
print("Classification Report:")
print(classification_report(y_test, logistic_preds))

# Save models
with open("vectorizer.pkl", "wb") as v:
    pickle.dump(vectorizer, v)
with open("logistic_model.pkl", "wb") as m:
    pickle.dump(logistic_model, m)

print("Model and vectorizer saved successfully!")
