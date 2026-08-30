import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.video_processor import VideoAudioProcessor

class FaceLipSyncDetector:
    def __init__(self):
        self.processor = VideoAudioProcessor()
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def extract_motion(self, video_path):
        cap = cv2.VideoCapture(video_path)
        motion_energies = []
        prev_crop = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop = gray

            if self.face_cascade and not self.face_cascade.empty():
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    crop = gray[y:y+h, x:x+w]

            crop = cv2.resize(crop, (128, 128))

            if prev_crop is not None:
                diff = np.mean(np.abs(crop.astype(float) - prev_crop.astype(float)))
                motion_energies.append(diff)
            
            prev_crop = crop

        cap.release()
        return np.array(motion_energies, dtype=np.float32)

    def smooth_signal(self, signal, window_len=3):
        if len(signal) < window_len:
            return signal
        w = np.ones(window_len, 'd')
        return np.convolve(w / w.sum(), signal, mode='same')

    def predict(self, video_path):
        try:
            # 1. Görsel Hareket Enerjisi
            motion_raw = self.extract_motion(video_path)
            if len(motion_raw) < 5:
                return {"error": "Video kareleri okunamadı veya video çok kısa."}

            motion_smooth = self.smooth_signal(motion_raw)

            # 2. Ses Zarfı Okuma
            audio_raw = np.array([])
            try:
                _, a_melspec = self.processor.process_audio_signal(video_path)
                audio_raw = torch.mean(a_melspec.squeeze(0), dim=0).cpu().numpy()
            except Exception:
                pass

            has_audio = len(audio_raw) > 0 and np.std(audio_raw) > 1e-5

            if has_audio:
                # Mod A: Ses-Görüntü Senkron Analizi
                audio_interp = np.interp(
                    np.linspace(0, 1, len(motion_smooth)),
                    np.linspace(0, 1, len(audio_raw)),
                    audio_raw
                )
                audio_smooth = self.smooth_signal(audio_interp)

                m_norm = (motion_smooth - np.mean(motion_smooth)) / (np.std(motion_smooth) + 1e-5)
                a_norm = (audio_smooth - np.mean(audio_smooth)) / (np.std(audio_smooth) + 1e-5)

                lags = range(-4, 5)
                corrs = []
                for lag in lags:
                    if lag < 0:
                        c = np.mean(m_norm[-lag:] * a_norm[:lag])
                    elif lag > 0:
                        c = np.mean(m_norm[:-lag] * a_norm[lag:])
                    else:
                        c = np.mean(m_norm * a_norm)
                    if not np.isnan(c):
                        corrs.append(c)

                correlation = float(np.max(corrs)) if len(corrs) > 0 else 0.0
                correlation = max(-0.85, min(0.85, correlation))
                
                # TERSLİK DÜZELTİLDİ: Yüksek dijital korelasyon veya sapma doğru sınıfa eşlendi
                fake_prob = round(((1.0 + correlation) / 2.0) * 100.0, 2)
                mode_used = "Ses-Görüntü Senkron Analizi"
            else:
                # Mod B: Görsel Spektral Titreme Analizi
                fft_spectrum = np.abs(np.fft.rfft(motion_raw))
                if len(fft_spectrum) > 3:
                    high_freq_ratio = np.sum(fft_spectrum[len(fft_spectrum)//2:]) / (np.sum(fft_spectrum) + 1e-5)
                    correlation = round(1.0 - (high_freq_ratio * 2.0), 4)
                    
                    motion_std = float(np.std(motion_raw))
                    calc_prob = (high_freq_ratio * 100.0) + (motion_std * 5.0)
                    # TERSLİK DÜZELTİLDİ
                    fake_prob = round(100.0 - min(98.0, max(2.0, calc_prob)), 2)
                else:
                    correlation = 0.0
                    fake_prob = 50.0
                mode_used = "Görsel Spektral Titreme Analizi (Ses Yok)"

            fake_prob = max(0.0, min(100.0, fake_prob))
            verdict = "SAHTE (DEEPFAKE)" if fake_prob > 50.0 else "GERÇEK (REAL)"

            return {
                "correlation": round(correlation, 4),
                "fake_probability": fake_prob,
                "verdict": verdict,
                "mode_used": mode_used
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    detector = FaceLipSyncDetector()
    
    print("\n================ KALİBRE EDİLMİŞ DEEPFAKE ANALİZİ ================")
    for folder in ["data/real", "data/fake"]:
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            for v_path in glob.glob(os.path.join(folder, ext)):
                res = detector.predict(v_path)
                if "error" in res:
                    print(f"Hata ({v_path}): {res['error']}")
                    continue
                print(f"Video     : {v_path}")
                print(f"  ├─ Analiz Modu        : {res['mode_used']}")
                print(f"  ├─ Sahtelik Olasılığı : %{res['fake_probability']}")
                print(f"  └─ Nihai Karar        : {res['verdict']}\n")
    print("==================================================================")