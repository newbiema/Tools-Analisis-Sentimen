import pandas as pd
import os

input_file = "hasil_labeling/hasil_labeling_indobert.csv"
output_file = "hasil_labeling/validasi_manual.xlsx"

df = pd.read_csv(input_file)

# ambil data confidence rendah
validasi = df[df["confidence"] < 0.70].copy()

# kalau terlalu sedikit, ambil sample tambahan 10%
sample_tambahan = df.sample(frac=0.10, random_state=42)

validasi = pd.concat([validasi, sample_tambahan])
validasi = validasi.drop_duplicates(subset=["komentar"])

# kolom yang dicek manual
validasi = validasi[[
    "komentar",
    "komentar_bersih",
    "label_indobert",
    "confidence",
    "label_final"
]]

validasi.to_excel(output_file, index=False)

print(f"Total data untuk validasi manual: {len(validasi)}")
print(f"File tersimpan di: {output_file}")