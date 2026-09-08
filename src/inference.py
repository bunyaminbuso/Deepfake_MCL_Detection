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
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.model_loaded = True
            except Exception:
                self.model_loaded = False

    def extract_video_frames(self, video_path, max_frames=30):
        """
        app.py uyumluluğu için SADECE VE SADECE tek bir NumPy dizisi (np.ndarray) döndürür.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((1, 96, 96, 3), dtype=np.float32)
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return np.zeros((1, 96, 96, 3), dtype=np.float32)

        step = max(1, total_frames // max_frames)
        frame_indices = [i * step for i in range(min(max_frames, total_frames))]
        
        raw_sampled_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                raw_sampled_frames.append(frame)
        cap.release()

        if not raw_sampled_frames:
            return np.zeros((1, 96, 96, 3), dtype=np.float32)

        face_boxes = []
        face_detected_count = 0
        
        if self.face_cascade is not None:
            for frame in raw_sampled_frames:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
                    
                    if len(faces) > 0:
                        best_face = max(faces, key=lambda b: b[2] * b[3])
                        face_boxes.append(best_face)
                        face_detected_count += 1
                    else:
                        face_boxes.append(None)
                except Exception:
                    face_boxes.append(None)
        else:
            face_boxes = [None] * len(raw_sampled_frames)

        self.last_has_face = (face_detected_count >= 1)

        last_valid_box = None
        for box in face_boxes:
            if box is not None:
                last_valid_box = box
                break

        face_crops = []
        full_crops = []
        for i, frame in enumerate(raw_sampled_frames):
            current_box = face_boxes[i] if face_boxes[i] is not None else last_valid_box
            H, W, _ = frame.shape
            
            if current_box is not None:
                x, y, w, h = current_box
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

        if not face_crops:
            return np.zeros((1, 96, 96, 3), dtype=np.float32)

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
            
            if self.model_loaded and self.model is not None and self.last_has_face:
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

                        model_prob = torch.sigmoid(outputs).item() * 100.0 if outputs.numel() == 1 else F.softmax(outputs, dim=1)[0][1].item() * 100.0
                        has_model_pred = True
                except Exception:
                    has_model_pred = False

            if has_model_pred:
                final_prob = round((model_prob * 0.40) + (artifact_score * 0.60), 2)
            else:
                final_prob = round(artifact_score, 2)

            verdict = "SAHTE (DEEPFAKE)" if final_prob >= 50.0 else "GERÇEK (REAL)"
            mode = "Piksel Tabanlı Spektral Artefakt Analizörü"

            return {
                "fake_probability": final_prob,
                "verdict": verdict,
                "mode_used": mode
            }
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}