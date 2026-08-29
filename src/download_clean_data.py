import os
import urllib.request

# Klasor yapisini hazirla
os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

# Sesi ve goruntusu %100 dogrulanmis acik kaynakli test videolari
dataset_urls = {
    "data/real/real_video1.mp4": "https://www.w3schools.com/html/mov_bbb.mp4",
    "data/fake/fake_video1.mp4": "https://www.w3schools.com/html/movie.mp4"
}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

print("--- TEMIZ VE SESLI VERI SETI INDIRILIYOR ---")
for path, url in dataset_urls.items():
    print(f"Indiriliyor: {path}...")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Tamamlandi: {path}")

print("\nVeri seti basariyla yenilendi!")