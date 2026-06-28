import pandas as pd

df = pd.read_csv("dataset_final_sentiment.csv")

df_negatif = df[df["label_final"] == "negatif"]
df_positif = df[df["label_final"] == "positif"]
df_netral = df[df["label_final"] == "netral"]

min_count = min(len(df_negatif), len(df_positif), len(df_netral))

df_negatif_sample = df_negatif.sample(min_count, random_state=42)
df_positif_sample = df_positif.sample(min_count, random_state=42)
df_netral_sample = df_netral.sample(min_count, random_state=42)

df_balanced = pd.concat([
    df_negatif_sample,
    df_positif_sample,
    df_netral_sample
])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced.to_csv("dataset_balanced_sentiment.csv", index=False, encoding="utf-8-sig")

print("Dataset balanced berhasil dibuat!")
print(df_balanced["label_final"].value_counts())