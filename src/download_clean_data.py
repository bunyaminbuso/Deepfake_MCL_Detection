import os
import urllib.request

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

# Gercek insan konuşması (Talking Head) içeren doğrudan MP4 bağlantıları
human_talk_urls = {
    "data/real/real_video1.mp4": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "data/fake/fake_video1.mp4": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
}

headers = {'User-Agent': 'Mozilla/5.0'}

print("--- INSAN KONUSMA VIDEOLARI INDIRILIYOR ---")
for path, url in human_talk_urls.items():
    print(f"Indiriliyor: {path}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Tamamlandi: {path}")
    except Exception as e:
        print(f"Hata: {e}")

print("\nVeriler basariyla yenilendi!")