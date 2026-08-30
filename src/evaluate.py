import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
from inference import FaceLipSyncDetector

def run_academic_evaluation():
    detector = FaceLipSyncDetector()
    
    # Karmaşıklık Matrisi (Confusion Matrix) Değişkenleri
    tp = 0  # True Positive: Sahte video doğru şekilde Sahte tahmin edildi
    fp = 0  # False Positive: Gerçek video hatalı şekilde Sahte tahmin edildi (Tip I Hata)
    tn = 0  # True Negative: Gerçek video doğru şekilde Gerçek tahmin edildi
    fn = 0  # False Negative: Sahte video hatalı şekilde Gerçek tahmin edildi (Tip II Hata)

    print("\n================ AKADEMİK DEĞERLENDİRME VE TEST RAPORU ================\n")
    
    for folder in ["data/real", "data/fake"]:
        is_actual_fake = (folder == "data/fake")
        expected_verdict = "SAHTE (DEEPFAKE)" if is_actual_fake else "GERÇEK (REAL)"
        
        videos = []
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            videos.extend(glob.glob(os.path.join(folder, ext)))
        
        for v_path in videos:
            res = detector.predict(v_path)
            if "error" in res:
                print(f"[HATA] {os.path.basename(v_path)}: {res['error']}")
                continue
            
            is_predicted_fake = ("SAHTE" in res['verdict'])
            
            if is_actual_fake and is_predicted_fake:
                tp += 1
                status = "DOĞRU (TP)"
            elif not is_actual_fake and is_predicted_fake:
                fp += 1
                status = "YANLIŞ (FP)"
            elif not is_actual_fake and not is_predicted_fake:
                tn += 1
                status = "DOĞRU (TN)"
            else:
                fn += 1
                status = "YANLIŞ (FN)"

            print(f"[{status}] {os.path.basename(v_path)} | Olasılık: %{res['fake_probability']} | Tahmin: {res['verdict']}")

    total = tp + fp + tn + fn
    if total == 0:
        print("\n[!] Test edilecek video bulunamadı. Lütfen 'data/real' ve 'data/fake' klasörlerini kontrol edin.")
        return

    # İstatistiksel Metrik Hesaplamaları
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "="*65)
    print("                  KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)")
    print("="*65)
    print(f"                       [ Tahmin: SAHTE ]   [ Tahmin: GERÇEK ]")
    print(f"  [ Gerçek: SAHTE  ]   TP = {tp:<13}   FN = {fn}")
    print(f"  [ Gerçek: GERÇEK ]   FP = {fp:<13}   TN = {tn}")
    print("="*65)
    print("                     AKADEMİK PERFORMANS METRİKLERİ")
    print("="*65)
    print(f"  Toplam Test Edilen Video : {total}")
    print(f"  Accuracy (Genel Doğruluk): %{accuracy * 100:.2f}")
    print(f"  Precision (Hassasiyet)   : %{precision * 100:.2f}")
    print(f"  Recall (Duyarlılık)      : %{recall * 100:.2f}")
    print(f"  F1-Score                 : %{f1_score * 100:.2f}")
    print("="*65 + "\n")

if __name__ == "__main__":
    run_academic_evaluation()