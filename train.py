# Import required libraries
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data.iloc[:, [0, 1]]  # Keeping only relevant columns
data.columns = ["label", "text"]
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split dataset
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Vectorization
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# Train Multiple Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

# Create 'models' folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Save models
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pickle.dump(model, open(f"models/{name.replace(' ', '_')}_model.pkl", "wb"))

# Save the vectorizer
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Models trained and saved in 'models/' folder.")
