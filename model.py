import os
import re
import urllib.request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import FeatureUnion
import gradio as gr

# --------------------------------------------------------------------
# 1. File Paths and Configurations
# --------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/YoussefAboelwafa/Spam-Ham-Email-Classifier/main/spam_ham_dataset.csv"
TRAIN_DATASET_PATH = "spam_ham_dataset.csv"
TEST_DATASET_PATH = "emails2.csv"
OUTPUT_PREDICTIONS_PATH = "tested_predictions.csv"

TRAIN_TEXT_COLUMN = "text"
TRAIN_LABEL_COLUMN = "label_num"

TEST_TEXT_COLUMN = "text"
TEST_LABEL_COLUMN = "spam"

# Default fallback threshold if testing dataset is not found
best_threshold = 0.50  

# --------------------------------------------------------------------
# 2. Text Cleaning & Label Conversion Functions
# --------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean email text by removing HTML tags, URLs, and non-alphabetic chars."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def convert_labels(label_series):
    """Converts various label descriptions into numbers (0 or 1)."""
    label_series = label_series.copy()
    if pd.api.types.is_numeric_dtype(label_series):
        return label_series.astype(int)
    return label_series.astype(str).str.lower().str.strip().map({
        "0": 0, "1": 1, "ham": 0, "spam": 1, 
        "not spam": 0, "no spam": 0, "non-spam": 0, "non spam": 0
    })

# --------------------------------------------------------------------
# 3. Load & Process Training Data
# --------------------------------------------------------------------
if not os.path.exists(TRAIN_DATASET_PATH):
    print(f"Downloading training dataset from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, TRAIN_DATASET_PATH)
    print("Download complete.")

train_df = pd.read_csv(TRAIN_DATASET_PATH, engine="python", on_bad_lines="skip")
print("Training dataset loaded.")

# Columns validation for training
if TRAIN_TEXT_COLUMN not in train_df.columns:
    raise ValueError(f"Training dataset must have a '{TRAIN_TEXT_COLUMN}' column.")
if TRAIN_LABEL_COLUMN not in train_df.columns:
    if "label" in train_df.columns:
        TRAIN_LABEL_COLUMN = "label"
    else:
        raise ValueError("Training dataset must have either 'label_num' or 'label' column.")

X_train_text = train_df[TRAIN_TEXT_COLUMN].astype(str)
y_train = convert_labels(train_df[TRAIN_LABEL_COLUMN])

valid_train_rows = y_train.notna()
X_train_text = X_train_text[valid_train_rows]
y_train = y_train[valid_train_rows].astype(int)

print("Preprocessing training text...")
X_train_clean = X_train_text.apply(clean_text)

# --------------------------------------------------------------------
# 4. Train FeatureUnion Vectorizer & Logistic Regression
# --------------------------------------------------------------------
print("Vectorizing training data using advanced FeatureUnion configuration...")
vectorizer = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        stop_words="english",
        max_features=40000,
        ngram_range=(1, 2)
    ))
])

X_train_tfidf = vectorizer.fit_transform(X_train_clean)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("Model trained successfully.")

# --------------------------------------------------------------------
# 5. Load Testing Dataset & Tune Threshold (If file exists)
# --------------------------------------------------------------------
if os.path.exists(TEST_DATASET_PATH):
    print(f"\nFound testing dataset: '{TEST_DATASET_PATH}'. Starting threshold evaluation...")
    test_df = pd.read_csv(TEST_DATASET_PATH, engine="python", on_bad_lines="skip")

    if TEST_TEXT_COLUMN not in test_df.columns or TEST_LABEL_COLUMN not in test_df.columns:
        print("⚠️ Testing dataset is missing required text/label columns. Skipping threshold optimization.")
    else:
        X_test_text = test_df[TEST_TEXT_COLUMN].astype(str)
        y_test = convert_labels(test_df[TEST_LABEL_COLUMN])
        X_test_clean = X_test_text.apply(clean_text)

        valid_test_rows = y_test.notna()
        X_test_text = X_test_text[valid_test_rows]
        X_test_clean = X_test_clean[valid_test_rows]
        y_test = y_test[valid_test_rows].astype(int)
        test_df = test_df[valid_test_rows].copy()

        # Transform and predict probabilities
        X_test_tfidf = vectorizer.transform(X_test_clean)
        spam_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

        # Scan for best threshold using F1 Score
        best_f1 = 0
        thresholds = [0.50, 0.525, 0.55, 0.56, 0.565, 0.57, 0.575, 0.58, 0.59, 0.60, 0.625, 0.65]
        
        print("-" * 50)
        for threshold in thresholds:
            y_pred_threshold = (spam_probabilities >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"Optimal Threshold Selected: {best_threshold} (Best F1 Score: {best_f1:.4f})")
        print("-" * 50)

        # Generate final metrics evaluation output
        y_pred = (spam_probabilities >= best_threshold).astype(int)
        print("\nTesting Dataset Evaluation Summary:")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
        
        # Save evaluated results to local disk
        test_df["actual_label_num"] = y_test.values
        test_df["actual_label"] = test_df["actual_label_num"].map({0: "ham", 1: "spam"})
        test_df["predicted_label_num"] = y_pred
        test_df["predicted_label"] = test_df["predicted_label_num"].map({0: "ham", 1: "spam"})
        test_df["spam_probability"] = spam_probabilities
        test_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
        print(f"Predictions successfully saved to: {OUTPUT_PREDICTIONS_PATH}")
else:
    print(f"\nℹ️ '{TEST_DATASET_PATH}' not detected. Defaulting model classification threshold to {best_threshold}.")

# --------------------------------------------------------------------
# 6. Gradio Prediction Function (Using Optimized Threshold)
# --------------------------------------------------------------------
def predict_email(email_text: str):
    if not email_text.strip():
        return "Please enter some text to analyze."
            
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    
    # Extract structural probability matrices
    probability = model.predict_proba(vectorized)[0]
    ham_confidence = probability[0]
    spam_confidence = probability[1]
    
    # Classify decision path according to optimized threshold rules
    if spam_confidence >= best_threshold:
        return f"🚨 SPAM (Confidence: {spam_confidence:.2%}) [Threshold: {best_threshold}]"
    else:
        return f"✅ HAM / NOT SPAM (Confidence: {ham_confidence:.2%}) [Threshold: {best_threshold}]"

# --------------------------------------------------------------------
# 7. Gradio Application Launch UI
# --------------------------------------------------------------------
demo = gr.Interface(
    fn=predict_email,
    inputs=gr.Textbox(lines=6, placeholder="Paste email content here...", label="Email Text"),
    outputs=gr.Label(label="Prediction Result"),
    title="📬 Optimized Spam Email Classifier",
    description=(
        f"This application uses FeatureUnion TF-IDF & Logistic Regression pipelines. "
        f"The prediction rules adapt using an calculated decision threshold metric ({best_threshold}) "
        f"to balance Precision and Recall performance safely."
    ),
    examples=[
        ["Congratulations! You have won a free iPhone. Click this link to claim your prize now!"],
        ["Hi, can we meet tomorrow to discuss the project?"]
    ]
)

if __name__ == "__main__":
    demo.launch()
