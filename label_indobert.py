import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os

input_file = "hasil_preprocessing/hasil_preprocessing_revisi.csv"
output_folder = "hasil_labeling"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_file)

classifier = pipeline(
    "text-classification",
    model="mdhugol/indonesia-bert-sentiment-classification",
    tokenizer="mdhugol/indonesia-bert-sentiment-classification"
)

def label_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "netral", 0

    result = classifier(str(text)[:512])[0]
    label = result["label"].lower()
    score = result["score"]

    if "positive" in label or "positif" in label:
        return "positif", score
    elif "negative" in label or "negatif" in label:
        return "negatif", score
    elif "neutral" in label or "netral" in label:
        return "netral", score
    else:
        return label, score

labels = []
scores = []

for text in tqdm(df["komentar_bersih"], desc="Labeling IndoBERT"):
    label, score = label_sentiment(text)
    labels.append(label)
    scores.append(score)

df["label_indobert"] = labels
df["confidence"] = scores
df["label_final"] = df["label_indobert"]

output_file = os.path.join(output_folder, "hasil_labeling_indobert.csv")
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Selesai!")
print(df["label_indobert"].value_counts())
print(f"File tersimpan di: {output_file}")