import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA FINAL
# =========================
df = pd.read_csv("dataset_final_sentiment.csv")

print("Jumlah data:", len(df))
print("\nDistribusi label:")
print(df["label_final"].value_counts())

# =========================
# BERSIHKAN DATA KOSONG
# =========================
df = df.dropna(subset=["komentar_bersih", "label_final"])

df["komentar_bersih"] = df["komentar_bersih"].astype(str)
df["label_final"] = df["label_final"].astype(str).str.lower().str.strip()

label_valid = ["positif", "negatif", "netral"]
df = df[df["label_final"].isin(label_valid)]

print("\nJumlah data setelah cleaning:", len(df))
print("\nDistribusi label setelah cleaning:")
print(df["label_final"].value_counts())

# =========================
# FITUR DAN LABEL
# =========================
X = df["komentar_bersih"]
y = df["label_final"]

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(X)

print("\nShape TF-IDF:", X_tfidf.shape)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah Training:", X_train.shape[0])
print("Jumlah Testing :", X_test.shape[0])

# =========================
# NAIVE BAYES
# =========================
model = MultinomialNB(alpha=0.5)

model.fit(X_train, y_train)

# =========================
# PREDIKSI
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUASI
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print("HASIL EVALUASI MODEL")
print("=========================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
labels = ["negatif", "netral", "positif"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Naive Bayes")
plt.show()