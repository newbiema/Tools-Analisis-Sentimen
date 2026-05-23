import pandas as pd
import os

# 📁 Folder tempat semua file CSV kamu
folder_path = "hasil_scrapping"

# ambil semua file CSV
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dataframes = []

for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# 🔥 Gabungkan semua data
combined_df = pd.concat(dataframes, ignore_index=True)

# 🔥 Hapus duplikat berdasarkan komentar
combined_df.drop_duplicates(subset=["komentar"], inplace=True)

# 🔥 Reset index
combined_df.reset_index(drop=True, inplace=True)

# 💾 Simpan hasil gabungan
output_path = os.path.join(folder_path, "gabungan_komentar.csv")
combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Total data setelah digabung: {len(combined_df)}")
print(f"File tersimpan di: {output_path}")
print(combined_df.head())