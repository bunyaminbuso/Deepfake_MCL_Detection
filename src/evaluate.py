import sys
import os
# Python'ın 'utils' ve diğer modülleri bulabilmesi için yol tanımı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
from inference import FaceLipSyncDetector

def run_evaluation():
    detector = FaceLipSyncDetector()
    
    correct = 0
    total = 0
    
    print("\n--- TOPLU PERFORMANS VE DEĞERLENDİRME RAPORU ---")
    
    for folder in ["data/real", "data/fake"]:
        expected = "GERÇEK (REAL)" if folder == "data/real" else "SAHTE (DEEPFAKE)"
        videos = []
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            videos.extend(glob.glob(os.path.join(folder, ext)))
        
        for v_path in videos:
            res = detector.predict(v_path)
            if "error" in res:
                continue
                
            is_correct = res['verdict'] == expected
            if is_correct:
                correct += 1
            total += 1
            
            status = "PASSED" if is_correct else "FAILED"
            print(f"[{status}] {os.path.basename(v_path)} -> Tahmin: {res['verdict']} | Beklenen: {expected}")

    accuracy = (correct / total * 100) if total > 0 else 0.0
    print("-" * 50)
    print(f"Toplam Test Edilen Video : {total}")
    print(f"Doğru Tahmin Sayısı     : {correct}")
    print(f"Sistem Başarı Oranı (Acc): %{accuracy:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    run_evaluation()