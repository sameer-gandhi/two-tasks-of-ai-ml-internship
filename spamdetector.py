# 🚀 Complete Human-Made Spam Detector with Interactive Mode
# This is the full standalone script with training + interactive mode + auto-save
# Created for Sameer Malik

import os
import json
import joblib
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

MODEL_FILE = "spam_model.pkl"
VEC_FILE = "vectorizer.pkl"
DATA_FILE = "spam_data.json"

# ---------------------------
# Load or Create Dataset
# ---------------------------
def load_dataset():
    if not os.path.exists(DATA_FILE):
        data = {
            "messages": [
                ("Congratulations! You won ₹50000 lottery", "spam"),
                ("Get free recharge now!!!", "spam"),
                ("Hello Sameer, your meeting is at 5 PM", "ham"),
                ("Your package has been shipped", "ham")
            ]
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
    else:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    return data["messages"]

# ---------------------------
# Train the Model
# ---------------------------
def train_model():
    dataset = load_dataset()
    msgs, labels = zip(*dataset)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(msgs)
    y = labels

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VEC_FILE)

    print("\n✅ Model Trained Successfully! Saved as spam_model.pkl")
    print("✅ Vectorizer Saved as vectorizer.pkl\n")

# ---------------------------
# Predict
# ---------------------------
def predict_spam(message):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VEC_FILE):
        print("❌ No trained model found! Please run: python spamdetector.py --train")
        return

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VEC_FILE)

    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]

    if prediction == "spam":
        print("🚨 SPAM DETECTED!")
    else:
        print("✔️ Message is Safe (HAM)")

# ---------------------------
# Interactive Mode
# ---------------------------
def interactive_mode():
    print("\n🎯 Interactive Mode Activated")
    print("Type a message to check if it's spam (type 'exit' to quit)\n")

    if not os.path.exists(MODEL_FILE):
        print("❌ No trained model found! Please run: python spamdetector.py --train")
        return

    while True:
        msg = input("Enter message: ")
        if msg.lower() == "exit":
            print("👋 Exiting interactive mode…")
            break
        predict_spam(msg)


# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Train the spam model")
parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
args = parser.parse_args()

if args.train:
    train_model()
elif args.interactive:
    interactive_mode()
else:
    print("\nUsage:")
    print(" python spamdetector.py --train        → Train the model")
    print(" python spamdetector.py --interactive  → Start chat mode\n")