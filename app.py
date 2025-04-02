from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load models from 'models/' folder
try:
    nb_model = pickle.load(open("models/Naive_Bayes_model.pkl", "rb"))
    lr_model = pickle.load(open("models/Logistic_Regression_model.pkl", "rb"))
    rf_model = pickle.load(open("models/Random_Forest_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    message_vec = vectorizer.transform([message])

    predictions = {
        "Naive Bayes": "Spam" if nb_model.predict(message_vec)[0] == 1 else "Ham",
        "Logistic Regression": "Spam" if lr_model.predict(message_vec)[0] == 1 else "Ham",
        "Random Forest": "Spam" if rf_model.predict(message_vec)[0] == 1 else "Ham"
    }

    return render_template("index.html", predictions=predictions, message=message)

if __name__ == "__main__":
    app.run(port=5001, debug=False)

