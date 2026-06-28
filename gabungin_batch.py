import pandas as pd
import os
import glob

folder_path = "hasil_labeling_manual"

files = glob.glob(os.path.join(folder_path, "batch_*_labeled.csv"))

dataframes = []

for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

df_final = pd.concat(dataframes, ignore_index=True)

# hapus duplikat kalau ada
df_final.drop_duplicates(subset=["komentar"], inplace=True)

# rapikan label
df_final["label_final"] = df_final["label_final"].str.lower().str.strip()

# hanya ambil label valid
label_valid = ["positif", "negatif", "netral"]
df_final = df_final[df_final["label_final"].isin(label_valid)]

df_final.reset_index(drop=True, inplace=True)

df_final.to_csv("dataset_final_sentiment.csv", index=False, encoding="utf-8-sig")

print("Dataset final berhasil dibuat!")
print("Jumlah data:", len(df_final))
print("\nDistribusi label:")
print(df_final["label_final"].value_counts())