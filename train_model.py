import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv("expenses_6months_200.csv")

# Features and target
X = df['Description']
y = df['Category']

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, "expense_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
