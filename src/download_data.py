import os
import cv2
import numpy as np

# Indirme/Veri klasorlerini kontrol et
os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

def generate_sample_video(filename, is_fake=False):
    width, height = 224, 224
    fps = 30
    duration_sec = 3
    total_frames = fps * duration_sec
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color = (0, 0, 255) if is_fake else (0, 255, 0)
        
        # Hareket simulasyonu
        center_x = int(width / 2 + 30 * np.sin(i / 5.0))
        center_y = int(height / 2 + 30 * np.cos(i / 5.0))
        cv2.circle(frame, (center_x, center_y), 40, color, -1)
        
        out.write(frame)
        
    out.release()
    print(f"Olusturuldu: {filename}")

print("Veri seti videolari olusturuluyor...")
generate_sample_video("data/real/real_sample2.mp4", is_fake=False)
generate_sample_video("data/fake/fake_sample2.mp4", is_fake=True)
print("Veri indirme ve hazirlama tamamlandi!")