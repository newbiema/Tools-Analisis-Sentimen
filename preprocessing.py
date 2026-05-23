import pandas as pd
import re
import string
import os
import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

nltk.download("stopwords")

# =========================
# LOAD DATA
# =========================
input_file = "hasil_scrapping/gabungan_komentar.csv"
output_folder = "hasil_preprocessing"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_file)
df = df[["komentar"]]
df.dropna(inplace=True)
df.drop_duplicates(subset=["komentar"], inplace=True)

# =========================
# SETUP STEMMER & STOPWORD
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words("indonesian"))

# Kata penting JANGAN DIHAPUS
important_words = {
    "timnas", "indonesia", "pssi", "sty", "patrick",
    "pelatih", "pemain", "bola", "dunia", "kualifikasi",
    "gagal", "menang", "kalah", "lolos", "main"
}

stop_words = stop_words - important_words

# Stopword tambahan yang benar-benar noise
custom_stopwords = {
    "yg", "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "aja", "saja", "sih", "dong", "deh", "lah", "nih",
    "wkwk", "wkwkwk", "haha", "hehe", "hahaha",
    "bro", "cuy", "min", "admin", "guys",
    "gue", "gua", "gw", "lu", "lo",
    "nya", "kok", "kayak", "kek", "banget",
    "pas", "tau", "tahu", "ish", "eh", "oh"
}

# pastikan kata penting tidak ikut custom stopword
custom_stopwords = custom_stopwords - important_words
stop_words.update(custom_stopwords)

# =========================
# NORMALISASI KATA GAUL
# =========================
normalization_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bangett": "banget",
    "bangettt": "banget",
    "krn": "karena",
    "karna": "karena",
    "sm": "sama",
    "sma": "sama",
    "utk": "untuk",
    "dgn": "dengan",
    "dlm": "dalam",
    "org": "orang",
    "pd": "pada",
    "trs": "terus",
    "trus": "terus",
    "jd": "jadi",
    "jdi": "jadi",
    "tp": "tapi",
    "tpi": "tapi",
    "klo": "kalau",
    "kl": "kalau",
    "dr": "dari",
    "knp": "kenapa",
    "mantab": "mantap",
    "mantapp": "mantap",
    "jelekknya": "jelek",
    "mainnya": "main"
}

def normalize_word(word):
    return normalization_dict.get(word, word)

# =========================
# CLEANSING
# =========================
def cleansing(text):
    text = str(text)

    # hapus URL
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # hapus mention dan hashtag, tapi kata hashtag bisa dipertahankan kalau mau
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)

    # hapus angka
    text = re.sub(r"\d+", " ", text)

    # hapus emoji / karakter non-ascii
    text = text.encode("ascii", "ignore").decode("ascii")

    # hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================
# CASE FOLDING
# =========================
def case_folding(text):
    return text.lower()

# =========================
# TOKENIZATION
# =========================
def tokenization(text):
    return text.split()

# =========================
# NORMALIZATION
# =========================
def normalization(tokens):
    return [normalize_word(word) for word in tokens]

# =========================
# STOPWORD REMOVAL
# =========================
def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

# =========================
# STEMMING
# =========================
def stemming(tokens):
    hasil = []
    for word in tokens:
        if word in important_words:
            hasil.append(word)
        else:
            hasil.append(stemmer.stem(word))
    return hasil

# =========================
# FILTERING
# =========================
def filtering(tokens):
    filtered = []

    for word in tokens:
        # jangan buang kata 2 huruf seperti "st", "id", dll
        if len(word) <= 1:
            continue

        # hapus token kosong
        if word.strip() == "":
            continue

        filtered.append(word)

    return filtered

# =========================
# PIPELINE
# =========================
def preprocess_text(text):
    text = cleansing(text)
    text = case_folding(text)
    tokens = tokenization(text)
    tokens = normalization(tokens)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    tokens = filtering(tokens)
    return " ".join(tokens)

df["komentar_bersih"] = df["komentar"].apply(preprocess_text)

# hapus hasil kosong
df = df[df["komentar_bersih"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

# simpan hasil
output_file = os.path.join(output_folder, "hasil_preprocessing_revisi.csv")
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("Preprocessing revisi selesai!")
print(f"Total data bersih: {len(df)}")
print(f"File tersimpan di: {output_file}")

# cek 10 hasil pertama
for i in range(min(10, len(df))):
    print("ASLI   :", df["komentar"].iloc[i])
    print("BERSIH :", df["komentar_bersih"].iloc[i])
    print("-" * 50)