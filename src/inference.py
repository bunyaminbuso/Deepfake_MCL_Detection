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
except Exception as e:
    print(f"[Bilgi] MCL Model modülü içe aktarılamadı: {e}")

class FaceLipSyncDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_cascade = None
        self.last_has_face = False
        self.last_full_frames = None

        try:
            cascade_file = 'haarcascade_frontalface_default.xml'
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                p = os.path.join(cv2.data.haarcascades, cascade_file)
                if os.path.exists(p):
                    self.face_cascade = cv2.CascadeClassifier(p)
            if self.face_cascade is None or self.face_cascade.empty():
                self.face_cascade = cv2.CascadeClassifier(cascade_file)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

        self.model = None
        self.model_loaded = False

        if MCLModelClass is not None:
            try:
                self.model = MCLModelClass().to(self.device)
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_mcl_model.pt'))
                
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # State Dict sarmalayıcılarını ayıkla
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint

                    # 'module.' öneklerini temizle
                    clean_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace('module.', '') if k.startswith('module.') else k
                        clean_state_dict[new_key] = v

                    self.model.load_state_dict(clean_state_dict, strict=False)
                    self.model.eval()
                    self.model_loaded = True
                    print("[BAŞARILI] Derin Öğrenme Modeli yüklendi ve aktif.")
                else:
                    print(f"[UYARI] Model checkpoint dosyası bulunamadı: {model_path}")
            except Exception as e:
                print(f"[HATA] Model yüklenirken hata oluştu: {e}")
                self.model_loaded = False

    def extract_video_frames(self, video_path, max_frames=12):
        """
        cap.grab() ile kare decode etmeden hızlı atlama ve tek kare yüz tespiti.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((1, 96, 96, 3), dtype=np.float32)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 120

        step = max(1, total_frames // max_frames)
        raw_sampled_frames = []
        
        curr_frame = 0
        while cap.isOpened() and len(raw_sampled_frames) < max_frames:
            if curr_frame % step == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                raw_sampled_frames.append(frame)
            else:
                ret = cap.grab()
                if not ret:
                    break
            curr_frame += 1

        cap.release()

        if not raw_sampled_frames:
            return np.zeros((1, 96, 96, 3), dtype=np.float32)

        best_box = None
        if self.face_cascade is not None:
            first_frame = raw_sampled_frames[0]
            H_orig, W_orig, _ = first_frame.shape
            scale = 320.0 / float(W_orig) if W_orig > 320 else 1.0

            small_first = cv2.resize(first_frame, (int(W_orig * scale), int(H_orig * scale))) if scale != 1.0 else first_frame
            gray_first = cv2.cvtColor(small_first, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray_first, scaleFactor=1.15, minNeighbors=3, minSize=(24, 24))
            if len(faces) > 0:
                b = max(faces, key=lambda x: x[2] * x[3])
                if scale != 1.0:
                    best_box = (int(b[0] / scale), int(b[1] / scale), int(b[2] / scale), int(b[3] / scale))
                else:
                    best_box = b

        self.last_has_face = (best_box is not None)

        face_crops = []
        full_crops = []
        for frame in raw_sampled_frames:
            H, W, _ = frame.shape
            if best_box is not None:
                x, y, w, h = best_box
                pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(W, x + w + pad_w), min(H, y + h + pad_h)
                crop_face = frame[y1:y2, x1:x2]
            else:
                crop_face = frame[int(H * 0.1):int(H * 0.8), int(W * 0.15):int(W * 0.85)]

            crop_full = frame[int(H * 0.08):int(H * 0.82), :]

            rgb_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
            rgb_full = cv2.cvtColor(crop_full, cv2.COLOR_BGR2RGB)

            face_crops.append(cv2.resize(rgb_face, (96, 96)))
            full_crops.append(cv2.resize(rgb_full, (96, 96)))

        face_arr = np.array(face_crops, dtype=np.float32) / 255.0
        self.last_full_frames = np.array(full_crops, dtype=np.float32) / 255.0

        return face_arr

    def _calculate_relative_artifact_score(self, face_frames, full_frames):
        if face_frames is None or len(face_frames) < 2:
            return 20.0

        if full_frames is None:
            full_frames = face_frames

        seam_scores = []
        chroma_scores = []
        for f in face_frames:
            img = (f * 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

            h, w = gray.shape
            inner_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(inner_mask, (w // 2, h // 2), (int(w * 0.3), int(h * 0.35)), 0, 0, 360, 255, -1)

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            inner_var = np.var(lap[inner_mask == 255]) + 1e-5
            outer_var = np.var(lap[inner_mask == 0]) + 1e-5

            seam_scores.append(outer_var / inner_var)

            cr_in = np.mean(ycrcb[:, :, 1][inner_mask == 255])
            cr_out = np.mean(ycrcb[:, :, 1][inner_mask == 0])
            chroma_scores.append(abs(cr_in - cr_out))

        mean_seam_ratio = float(np.mean(seam_scores))
        mean_chroma_delta = float(np.mean(chroma_scores))

        face_diff = np.abs(np.diff(face_frames, axis=0))
        full_diff = np.abs(np.diff(full_frames, axis=0))

        face_m = float(np.mean(face_diff)) + 1e-6
        full_m = float(np.mean(full_diff)) + 1e-6
        motion_ratio = face_m / full_m

        fft_diffs = []
        for f in face_frames:
            gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            f_shift = np.fft.fftshift(np.fft.fft2(gray))
            mag = 20 * np.log(np.abs(f_shift) + 1e-8)
            fft_diffs.append(np.std(mag))

        fft_std = float(np.std(fft_diffs))

        z = (
            (mean_seam_ratio - 1.15) * 4.0 +
            (motion_ratio - 1.05) * 5.0 +
            (mean_chroma_delta - 2.8) * 0.4 +
            (fft_std - 1.9) * 2.5
        )

        prob = 1.0 / (1.0 + np.exp(-z)) * 100.0
        return float(np.clip(prob, 5.0, 95.0))

    def predict(self, video_path):
        try:
            face_frames = self.extract_video_frames(video_path)
            if face_frames is None or len(face_frames) == 0:
                return {"error": "Video veya kare okunamadı."}

            full_frames = getattr(self, 'last_full_frames', face_frames)
            artifact_score = self._calculate_relative_artifact_score(face_frames, full_frames)

            model_prob = 0.0
            has_model_pred = False

            if self.model_loaded and self.model is not None:
                try:
                    with torch.no_grad():
                        v_tensor = torch.tensor(np.transpose(face_frames, (3, 0, 1, 2))).unsqueeze(0).float().to(self.device)
                        a_tensor = torch.zeros((1, 1, 80, 100), device=self.device)

                        try:
                            outputs = self.model(v_tensor, a_tensor)
                        except Exception:
                            outputs = self.model(v_tensor)

                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]

                        if outputs.numel() == 1:
                            model_prob = torch.sigmoid(outputs).item() * 100.0
                        else:
                            model_prob = F.softmax(outputs, dim=1)[0][1].item() * 100.0
                        
                        has_model_pred = True
                except Exception as e:
                    print(f"[UYARI] Derin öğrenme çıkarım hatası: {e}")
                    has_model_pred = False

            if has_model_pred:
                final_prob = round((model_prob * 0.70) + (artifact_score * 0.30), 2)
                mode = "MCL Derin Öğrenme Modeli + Spektral Analiz"
            else:
                final_prob = round(artifact_score, 2)
                mode = "Piksel Tabanlı Spektral Artefakt Analizörü (Yedek Mod)"

            verdict = "SAHTE (DEEPFAKE)" if final_prob >= 50.0 else "GERÇEK (REAL)"

            return {
                "fake_probability": final_prob,
                "verdict": verdict,
                "mode_used": mode,
                "model_active": has_model_pred
            }
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}