import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASETS
# ==========================================

print("Loading datasets...")

fake_data = pd.read_csv("data/Fake.csv")
real_data = pd.read_csv("data/True.csv")

print("Fake dataset loaded!")
print("Real dataset loaded!")
print()


# ==========================================
# 2. ADD LABELS
# ==========================================

fake_data["label"] = "FAKE"
real_data["label"] = "REAL"


# ==========================================
# 3. COMBINE DATASETS
# ==========================================

data = pd.concat(
    [fake_data, real_data],
    ignore_index=True
)

print("Datasets combined successfully!")
print("Total records:", len(data))
print()


# ==========================================
# 4. CREATE NEWS TEXT
# ==========================================

data["text"] = (
    data["title"].fillna("") +
    " " +
    data["text"].fillna("")
)


# ==========================================
# 5. REMOVE EMPTY DATA
# ==========================================

data = data[
    data["text"].str.strip() != ""
]

print("Data cleaning completed!")
print()


# ==========================================
# 6. INPUT AND OUTPUT
# ==========================================

X = data["text"]
y = data["label"]


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print()


# ==========================================
# 8. TF-IDF VECTORIZATION
# ==========================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("TF-IDF completed!")
print()


# ==========================================
# 9. CREATE MACHINE LEARNING MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("Training Machine Learning model...")

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed!")
print()


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 12. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("================================")
print("       MODEL EVALUATION")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()


# ==========================================
# 13. CLASSIFICATION REPORT
# ==========================================

print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["FAKE", "REAL"]
)

print("Confusion Matrix:")
print()

print(cm)

print()


# ==========================================
# 15. CREATE MODEL FOLDER
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)


# ==========================================
# 16. CONFUSION MATRIX GRAPH
# ==========================================
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["FAKE", "REAL"]
)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Fake News Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.yticks(
    [0, 1],
    ["FAKE", "REAL"]
)


# Add numbers inside matrix

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.colorbar()

plt.tight_layout()


# Save graph

plt.savefig(
    "model/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix graph saved!")
print(
    "model/confusion_matrix.png"
)

print()


# ==========================================
# 17. SAVE MACHINE LEARNING MODEL
# ==========================================

with open(
    "model/fake_news_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ==========================================
# 18. SAVE TF-IDF VECTORIZER
# ==========================================

with open(
    "model/tfidf_vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ==========================================
# 19. FINAL MESSAGE
# ==========================================

print("================================")
print("     FILES SAVED SUCCESSFULLY")
print("================================")

print()
print("1. Machine Learning Model:")
print("   model/fake_news_model.pkl")

print()

print("2. TF-IDF Vectorizer:")
print("   model/tfidf_vectorizer.pkl")

print()

print("3. Confusion Matrix:")
print("   model/confusion_matrix.png")

print()

print("================================")
print("       PROJECT TRAINING DONE")
print("================================")