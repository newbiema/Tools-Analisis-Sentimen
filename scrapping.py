from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
import re
import os

VIDEO_URL = input("Masukkan URL video YouTube: ")
MAX_COMMENTS = 1000

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()
 
video_id = VIDEO_URL.split("v=")[-1]


folder_path = "hasil_scrapping"
os.makedirs(folder_path, exist_ok=True)

downloader = YoutubeCommentDownloader()
comments = downloader.get_comments_from_url(VIDEO_URL, sort_by=0)

data = []

for i, comment in enumerate(comments):
    if i >= MAX_COMMENTS:
        break

    data.append({
        "id_komentar": comment.get("cid", ""),
        "tanggal": comment.get("time", ""),
        "komentar": clean_text(comment.get("text", "")),
        "like": comment.get("votes", 0),
        "jumlah_reply": comment.get("reply", 0),
        "video_url": VIDEO_URL
    })

df = pd.DataFrame(data)

# Simpan ke folder
file_path = os.path.join(folder_path, f"{video_id}.csv")
df.to_csv(file_path, index=False, encoding="utf-8-sig")

print(f"Berhasil mengambil {len(df)} komentar")
print(f"File tersimpan di: {file_path}")
print(df.head())