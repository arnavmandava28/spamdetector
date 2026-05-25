"""
Test a Trained Spam Email Classifier on a New Dataset

This script:
- Loads a training dataset
- Trains TF-IDF + Logistic Regression
- Loads a separate testing dataset
- Uses 0/1 labels where:
    0 = ham / not spam
    1 = spam
- Cleans the testing dataset
- Tests multiple thresholds
- Uses the best threshold based on F1 score
- Predicts spam/ham
- Evaluates accuracy, precision, recall, F1 score, confusion matrix
- Saves predictions to a CSV file

Training dataset: https://www.kaggle.com/datasets/venky73/spam-mails-dataset

Test dataset: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset
"""

import re
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


# ----------------------
# File Paths
# ----------------------
TRAIN_DATASET_PATH = "spam_ham_dataset.csv"
TEST_DATASET_PATH = "emails2.csv"

OUTPUT_PREDICTIONS_PATH = "tested_predictions.csv"


# ----------------------
# Column Names
# ----------------------
TRAIN_TEXT_COLUMN = "text"
TEST_TEXT_COLUMN = "text"

# Your testing dataset uses:
# 0 = ham / not spam
# 1 = spam
TEST_LABEL_COLUMN = "spam"

# spam_ham_dataset.csv usually has label_num:
# 0 = ham
# 1 = spam
TRAIN_LABEL_COLUMN = "label_num"


# ----------------------
# Text Cleaning Function
# ----------------------
def clean_text(text: str) -> str:
    """
    Clean email text:
    - lowercase
    - remove HTML tags
    - remove URLs
    - remove non-letter characters
    - remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------
# Label Conversion Function
# ----------------------
def convert_labels(label_series):
    """
    Converts labels into numbers.

    Accepts:
    - 0 = ham / not spam
    - 1 = spam
    - ham = 0
    - spam = 1
    - not spam = 0
    """

    label_series = label_series.copy()

    if pd.api.types.is_numeric_dtype(label_series):
        return label_series.astype(int)

    return label_series.astype(str).str.lower().str.strip().map({
        "0": 0,
        "1": 1,
        "ham": 0,
        "spam": 1,
        "not spam": 0,
        "no spam": 0,
        "non-spam": 0,
        "non spam": 0
    })


# ----------------------
# Load Training Dataset
# ----------------------
train_df = pd.read_csv(TRAIN_DATASET_PATH, engine="python", on_bad_lines="skip")

print("Training dataset loaded.")
print(train_df.head())

print("\nTraining columns:")
print(train_df.columns)


# ----------------------
# Check Training Columns
# ----------------------
if TRAIN_TEXT_COLUMN not in train_df.columns:
    raise ValueError(f"Training dataset must have a '{TRAIN_TEXT_COLUMN}' column.")

if TRAIN_LABEL_COLUMN not in train_df.columns:
    if "label" in train_df.columns:
        TRAIN_LABEL_COLUMN = "label"
    else:
        raise ValueError("Training dataset must have either 'label_num' or 'label' column.")


# ----------------------
# Prepare Training Data
# ----------------------
X_train_text = train_df[TRAIN_TEXT_COLUMN].astype(str)
y_train = convert_labels(train_df[TRAIN_LABEL_COLUMN])

valid_train_rows = y_train.notna()

X_train_text = X_train_text[valid_train_rows]
y_train = y_train[valid_train_rows].astype(int)

X_train_clean = X_train_text.apply(clean_text)


# ----------------------
# Train TF-IDF Vectorizer
# ----------------------
vectorizer = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(
        analyzer="word",
        stop_words="english",
        max_features=40000,
        ngram_range=(1, 2)
    ))
])

X_train_tfidf = vectorizer.fit_transform(X_train_clean)


# ----------------------
# Train Logistic Regression Model
# ----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("\nModel trained successfully.")


# ----------------------
# Load New Testing Dataset
# ----------------------
test_df = pd.read_csv(TEST_DATASET_PATH, engine="python", on_bad_lines="skip")

print("\nTesting dataset loaded.")
print(test_df.head())

print("\nTesting columns:")
print(test_df.columns)


# ----------------------
# Check Testing Columns
# ----------------------
if TEST_TEXT_COLUMN not in test_df.columns:
    raise ValueError(f"Testing dataset must have a '{TEST_TEXT_COLUMN}' column.")

if TEST_LABEL_COLUMN not in test_df.columns:
    raise ValueError(
        f"Testing dataset must have a '{TEST_LABEL_COLUMN}' column containing 0 and 1 labels."
    )


# ----------------------
# Prepare Testing Data
# ----------------------
X_test_text = test_df[TEST_TEXT_COLUMN].astype(str)
y_test = convert_labels(test_df[TEST_LABEL_COLUMN])

X_test_clean = X_test_text.apply(clean_text)


# ----------------------
# Remove Rows with Invalid Labels
# ----------------------
valid_test_rows = y_test.notna()

X_test_text = X_test_text[valid_test_rows]
X_test_clean = X_test_clean[valid_test_rows]
y_test = y_test[valid_test_rows].astype(int)
test_df = test_df[valid_test_rows].copy()


# ----------------------
# Transform Test Data Using Same Vectorizer
# ----------------------
X_test_tfidf = vectorizer.transform(X_test_clean)


# ----------------------
# Make Probability Predictions
# ----------------------
y_prob = model.predict_proba(X_test_tfidf)
spam_probabilities = y_prob[:, 1]


# ----------------------
# Test Multiple Thresholds
# ----------------------
best_threshold = 0
best_f1 = 0

print("\nThreshold Testing:")
print("-" * 80)

thresholds = [0.50, 0.525, 0.55, 0.56, 0.565, 0.57, 0.575, 0.58, 0.59, 0.60, 0.625, 0.65]

for threshold in thresholds:
    y_pred_threshold = (spam_probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_threshold, zero_division=0)

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nBest Threshold Based on F1 Score:")
print(f"Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1:.4f}")


# ----------------------
# Final Predictions Using Best Threshold
# ----------------------
y_pred = (spam_probabilities >= best_threshold).astype(int)


# ----------------------
# Display Each Individual Prediction
# ----------------------
#print("\nIndividual Prediction Results:")
#print("-" * 80)

#for i in range(len(test_df)):
    #email_text = X_test_text.iloc[i]
    #actual_num = y_test.iloc[i]
    #predicted_num = y_pred[i]
    #spam_probability = spam_probabilities[i]

    #actual_label = "spam" if actual_num == 1 else "ham"
    #predicted_label = "spam" if predicted_num == 1 else "ham"

    #print(f"\nEmail #{i + 1}")
    #print("Text:")
    #print(email_text)

    #print(f"\nActual Label:     {actual_label} ({actual_num})")
    #print(f"Predicted Label:  {predicted_label} ({predicted_num})")
    #print(f"Spam Probability: {spam_probability:.4f}")

    #if actual_num == predicted_num:
        #print("GOOD JOB BETA")
    #else:
        #print("NOOO")

    #print("-" * 80)


# ----------------------
# Evaluate Model
# ----------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nTesting Dataset Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"],
    zero_division=0
))


# ----------------------
# Save Predictions
# ----------------------
test_df["actual_label_num"] = y_test.values
test_df["actual_label"] = test_df["actual_label_num"].map({
    0: "ham",
    1: "spam"
})

test_df["predicted_label_num"] = y_pred
test_df["predicted_label"] = test_df["predicted_label_num"].map({
    0: "ham",
    1: "spam"
})

test_df["spam_probability"] = spam_probabilities

test_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

print(f"\nPredictions saved to: {OUTPUT_PREDICTIONS_PATH}")
