import pandas as pd
import os

df = pd.read_csv("hasil_preprocessing/hasil_preprocessing_revisi.csv")

batch_size = 50

os.makedirs("batch_labeling", exist_ok=True)

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]

    batch.to_csv(
        f"batch_labeling/batch_{i//batch_size+1}.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("Selesai membuat batch.")