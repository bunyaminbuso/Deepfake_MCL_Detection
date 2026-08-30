import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_dir = os.path.join(BASE_DIR, "data", "real")
fake_dir = os.path.join(BASE_DIR, "data", "fake")

print("\n🧹 Sentetik çizim videoları klasörlerden siliniyor...\n")

for folder in [real_dir, fake_dir]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.startswith("unique_") or f.startswith("synthetic_"):
                os.remove(os.path.join(folder, f))
                print(f"  [SİLİNDİ] -> {f}")

print("\n Klasörler temizlendi! Sadece gerçek videolar kaldı.")