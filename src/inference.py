import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect

MCLModelClass = None
try:
    import models.mcl_model as mcl_module
    for name, obj in inspect.getmembers(mcl_module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            MCLModelClass = obj
            break
except Exception:
    pass

class FaceLipSyncDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_cascade = None
        
        try:
            if hasattr(cv2, 'CascadeClassifier'):
                cascade_path = getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else ''
                if cascade_path and os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                else:
                    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    self.face_cascade = None
        except Exception:
            self.face_cascade = None

        self.model = None
        self.model_loaded = False
        self.working_shape_config = None

        if MCLModelClass is not None:
            try:
                self.model = MCLModelClass().to(self.device)
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_mcl_model.pt'))
                if os.path.exists(model_path):
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.model_loaded = True
            except Exception:
                self.model_loaded = False

    def extract_video_frames(self, video_path, max_frames=30):
        cap = cv2.VideoCapture(video_path)
        frames = []
        last_face_box = None

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Uzak yüzleri ve küçük kadrajları yakalamak için hassaslaştırılmış tespit
            if self.face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
                    if len(faces) > 0:
                        last_face_box = max(faces, key=lambda b: b[2] * b[3])
                except Exception:
                    pass

            if last_face_box is not None:
                x, y, w, h = last_face_box
                # Yüz çevresine %20 dinamik pay ekle
                pad_w, pad_h = int(w * 0.2), int(h * 0.2)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(frame.shape[1], x + w + pad_w), min(frame.shape[0], y + h + pad_h)
                face_crop = frame[y1:y2, x1:x2]
            else:
                # Yüz tespit edilemezse üst-orta gövde ve baş bölgesini odakla
                h_img, w_img = frame.shape[:2]
                crop_h, crop_w = int(h_img * 0.65), int(w_img * 0.65)
                start_x = (w_img - crop_w) // 2
                start_y = int(h_img * 0.05)
                face_crop = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]

            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(rgb, (96, 96))
            frames.append(crop)
            
        cap.release()
        if len(frames) == 0:
            return None
        return np.array(frames, dtype=np.float32) / 255.0

    def _calculate_artifact_score(self, raw_frames):
        # 1. Zamansal Hareket Değişkenliği (Temporal Variance)
        diffs = np.diff(raw_frames, axis=0)
        temporal_std = float(np.std(diffs))
        
        # 2. Kenar Keskinliği Tutarsızlığı (Laplacian Variance)
        lap_vars = [cv2.Laplacian((f * 255).astype(np.uint8), cv2.CV_64F).var() for f in raw_frames]
        lap_std = float(np.std(lap_vars))

        # 3. Yüzey Frekans Bozulması Analizi (FFT - Fast Fourier Transform)
        fft_scores = []
        for f in raw_frames:
            gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            f_shift = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = 20 * np.log(np.abs(f_shift) + 1e-8)
            fft_scores.append(np.std(magnitude))
        fft_std = float(np.std(fft_scores))
        
        # Çok Katmanlı Artefakt & Frekans Puanı Hesaplama
        artifact_prob = (temporal_std * 250.0) + (lap_std * 0.45) + (fft_std * 16.0)
        return min(98.5, max(5.0, artifact_prob))

    def predict(self, video_path):
        try:
            raw_frames = self.extract_video_frames(video_path)
            if raw_frames is None:
                return {"error": "Video veya kare okunamadı."}

            artifact_score = self._calculate_artifact_score(raw_frames)

            if self.model_loaded and self.model is not None:
                try:
                    with torch.no_grad():
                        T, H, W, C = raw_frames.shape
                        v_tensor = torch.tensor(np.transpose(raw_frames, (3, 0, 1, 2))).unsqueeze(0).float().to(self.device)
                        a_tensor = torch.zeros((1, 1, 80, 100), device=self.device)
                        
                        try:
                            outputs = self.model(v_tensor, a_tensor)
                        except Exception:
                            outputs = self.model(v_tensor)

                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]

                        raw_prob = torch.sigmoid(outputs).item() * 100.0 if outputs.numel() == 1 else F.softmax(outputs, dim=1)[0][1].item() * 100.0
                        
                        # Model ile Sinyal Artefaktını Hibrit Harmanlama
                        final_prob = round((raw_prob * 0.3) + (artifact_score * 0.7), 2)
                        
                        return {
                            "fake_probability": final_prob,
                            "verdict": "SAHTE (DEEPFAKE)" if final_prob > 38.0 else "GERÇEK (REAL)",
                            "mode_used": "Hibrit Derin Öğrenme & Dokusal Artefakt Analizörü"
                        }
                except Exception:
                    pass

            # Model devre dışıysa sadece Sinyal Motoru
            final_prob = round(artifact_score, 2)
            return {
                "fake_probability": final_prob,
                "verdict": "SAHTE (DEEPFAKE)" if final_prob > 38.0 else "GERÇEK (REAL)",
                "mode_used": "Gelişmiş Dokusal & Zamansal Sinyal Analizörü"
            }
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}