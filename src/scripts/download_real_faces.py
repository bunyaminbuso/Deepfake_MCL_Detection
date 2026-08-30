import os
import urllib.request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_dir = os.path.join(BASE_DIR, "data", "real")
fake_dir = os.path.join(BASE_DIR, "data", "fake")

# 1. ESKİ KLİPLERİ VE ÇÖPLERİ TEMİZLE
print("\n🧹 Eski bölünen klipler temizleniyor...")
for folder in [real_dir, fake_dir]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if any(k in f for k in ["_clip_", "synthetic_", "unique_", "copy_"]):
                try:
                    os.remove(os.path.join(folder, f))
                    print(f"  [SİLİNDİ] -> {f}")
                except Exception:
                    pass

# 2. İNTERNETTEN FARKLI GERÇEK İNSAN YÜZÜ VİDEOLARI İNDİR (Intel Public Sample Videos)
SAMPLE_VIDEOS = {
    "real": [
        ("human_female_face.mp4", "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4"),
        ("human_male_face.mp4", "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-male.mp4")
    ],
    "fake": [
        ("human_walk_face.mp4", "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4")
    ]
}

def download_file(url, target_path):
    print(f" ⬇️ İndiriliyor: {os.path.basename(target_path)}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"  [BAŞARILI] {os.path.basename(target_path)} indirildi.")

if __name__ == "__main__":
    print("\n🌐 Açık kaynaklı farklı insan yüzü videoları çekiliyor...\n")
    
    for category, items in SAMPLE_VIDEOS.items():
        target_dir = real_dir if category == "real" else fake_dir
        for filename, url in items:
            save_path = os.path.join(target_dir, filename)
            if not os.path.exists(save_path):
                try:
                    download_file(url, save_path)
                except Exception as e:
                    print(f"❌ İndirme hatası ({filename}): {e}")

    print("\n Yeni ve tamamen farklı insan yüzü videoları indirildi!")