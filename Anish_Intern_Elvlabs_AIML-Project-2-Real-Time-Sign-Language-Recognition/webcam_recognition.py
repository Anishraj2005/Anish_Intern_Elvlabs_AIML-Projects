"""
webcam_recognition.py
=======================
Real-time ASL letter recognition from webcam feed using a trained model and MediaPipe hand landmark detection.
This script captures video from the webcam, detects hand landmarks, predicts ASL letters, and builds sentences based on the recognized letters. 
It also provides a text-to-speech feature to read out the constructed sentence.
"""

#-------------- Import Setup -------------------
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

#---- Tracks whether TTS is currently playing so we don't overlap two calls.----
_tts_speaking = False

# ------------------------------Text-to-Speech Function------------------------------
def speak_text(text):
    if not _TTS_AVAILABLE:
        print("pyttsx3 not installed — TTS unavailable.")
        return

    global _tts_speaking
    if _tts_speaking:
        # Already speaking; ignore the extra press rather than overlap / crash.
        print("(TTS: still speaking, ignoring duplicate T press)")
        return

    # ------------------------- Run text-to-speech to avoid overlap ------------------------------
    def _speak():
        global _tts_speaking
        _tts_speaking = True
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)   # ~150 wpm; default 200 is too fast
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print(f"TTS error: {exc}")
        finally:
            _tts_speaking = False

    threading.Thread(target=_speak, daemon=True).start()
    print(f'Speaking: "{text}"')

# -------------------------------- Hand Landmark and Prediction Utilities ------------------------------
NUM_LANDMARKS, NUM_COORDS = 21, 3
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# -------------------------------- Landmark Preprocessing: Normalization and Flattening  ------------------------------
def normalize_landmarks(landmarks):
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)
    centered = landmarks - landmarks[0]
    scale = np.linalg.norm(centered, axis=1).max()
    return centered / scale if scale > 1e-8 else centered

# -------------------------------- Flattening Landmarks for Model Input ------------------------------
def flatten(landmarks_21x3):
    return np.asarray(landmarks_21x3, dtype=np.float32).reshape(-1)

# ------------------------------ Convert MediaPipe Result to NumPy Array ------------------------------
def mediapipe_result_to_array(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

#-------------- Letter Stabilizer Class -------------------
class LetterStabilizer:
    def __init__(self, hold_frames=15, confidence_threshold=0.75, release_frames=8):
        self.hold_frames, self.confidence_threshold, self.release_frames = (
            hold_frames, confidence_threshold, release_frames)
        self._current_label, self._current_count = None, 0
        self._since_release, self._last_committed, self._armed = 0, None, True

    # ------------------------------ Update Stabilizer State ------------------------------
    def update(self, label, confidence):
        if label is None or confidence < self.confidence_threshold:
            self._current_label, self._current_count = None, 0
            self._since_release += 1
            if self._since_release >= self.release_frames:
                self._armed = True
            return None
        if label == self._current_label:
            self._current_count += 1
        else:
            self._current_label, self._current_count = label, 1
        if label != self._last_committed:
            self._since_release += 1
            if self._since_release >= self.release_frames:
                self._armed = True
        else:
            self._since_release = 0
        if self._current_count >= self.hold_frames and self._armed:
            self._last_committed, self._armed = label, False
            self._since_release, self._current_count = 0, 0
            return label
        return None
    
    #------------------------------ Properties for Progress and Candidate ------------------------------ 
    @property
    def progress(self):
        return 0.0 if self._current_label is None else min(self._current_count / self.hold_frames, 1.0)

    @property
    def candidate(self):
        return self._current_label

#-------------- Sentence Builder Class -------------------
class SentenceBuilder:
    def __init__(self, hold_frames=15, confidence_threshold=0.75, release_frames=8, space_after_frames=45):
        self.stabilizer = LetterStabilizer(hold_frames, confidence_threshold, release_frames)
        self.space_after_frames = space_after_frames
        self._no_hand_count, self._space_pending_reset, self.text = 0, False, ""

    # ------------------------------ Update Sentence Based on Stabilizer Output ------------------------------
    def update(self, label, confidence, hand_present):
        committed = self.stabilizer.update(label, confidence)
        if committed:
            self.text += committed
            self._no_hand_count, self._space_pending_reset = 0, False
            return committed
        if hand_present:
            self._no_hand_count, self._space_pending_reset = 0, False
        else:
            self._no_hand_count += 1
            if (self._no_hand_count >= self.space_after_frames and not self._space_pending_reset
                    and self.text and not self.text.endswith(" ")):
                self.text += " "
                self._space_pending_reset = True
        return None
    
    #------------------------------ Backspace and Clear Functions ------------------------------ 
    def backspace(self):
        self.text = self.text[:-1]

    # ------------------------------ Clear Function ------------------------------
    def clear(self):
        self.text = ""
        self.stabilizer = LetterStabilizer(self.stabilizer.hold_frames,
                                            self.stabilizer.confidence_threshold,
                                            self.stabilizer.release_frames)
        self._no_hand_count, self._space_pending_reset = 0, False

#-------------- Letter Prediction Model -------------------
class LetterPredictor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / "label_classes.json") as f:
            self.classes = json.load(f)
        keras_path, sk_path = model_dir / "asl_cnn_model.keras", model_dir / "asl_landmark_model.pkl"
        if keras_path.exists():
            import tensorflow as tf
            self.backend, self.model = "keras", tf.keras.models.load_model(keras_path)
        elif sk_path.exists():
            import joblib
            self.backend = "sklearn"
            self.model = joblib.load(sk_path)
            self.scaler = joblib.load(model_dir / "scaler.pkl")
        else:
            raise FileNotFoundError(f"No trained model found in {model_dir}.")

    # ------------------------------ Predict Function ------------------------------
    def predict(self, landmarks_21x3):
        feat = flatten(normalize_landmarks(landmarks_21x3))
        if self.backend == "keras":
            probs = self.model.predict(feat.reshape(1, 21, 3), verbose=0)[0]
        else:
            probs = self.model.predict_proba(self.scaler.transform(feat.reshape(1, -1)))[0]
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx]), probs

#-------------- MediaPipe Hand Landmarker -------------------
def make_hand_landmarker(task_path, num_hands=1):
    BaseOptions, HandLandmarker = mp.tasks.BaseOptions, mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions, RunningMode = mp.tasks.vision.HandLandmarkerOptions, mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=RunningMode.VIDEO, num_hands=num_hands,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.6)
    return HandLandmarker.create_from_options(options)

#-------------- Drawing Utilities -------------------
def draw_skeleton(frame, landmarks_21x3):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks_21x3]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 120, 255), -1)

# ------------------------------ Draw UI Overlay ------------------------------
def draw_ui(frame, candidate, confidence, progress, sentence_text):
    h, w = frame.shape[:2]
    if candidate:
        cv2.putText(frame, f"{candidate}  {confidence:.0%}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        bar_w, filled = 220, int(220 * progress)
        cv2.rectangle(frame, (20, 65), (20 + bar_w, 80), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 65), (20 + filled, 80), (0, 255, 255), -1)
    else:
        cv2.putText(frame, "no hand", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, sentence_text + "_", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "BACKSPACE=del  C=clear  T=speak  Q=quit", (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

#-------------- Main Application Loop -------------------
def main(model_dir, task_path, camera_index, hold_frames, confidence_threshold):
    predictor = LetterPredictor(model_dir)
    landmarker = make_hand_landmarker(task_path)
    builder = SentenceBuilder(hold_frames=hold_frames, confidence_threshold=confidence_threshold,
                               release_frames=8, space_after_frames=45)
    if not _TTS_AVAILABLE:
        print("(pyttsx3 not installed -- text-to-speech disabled.)")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    start_time = time.time()
    print("Webcam recognition running. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        label, confidence, hand_present = None, 0.0, False
        if result.hand_landmarks:
            hand_present = True
            arr = mediapipe_result_to_array(result.hand_landmarks[0])
            label, confidence, _ = predictor.predict(arr)
            draw_skeleton(frame, arr)
            if confidence < confidence_threshold:
                label = None

        builder.update(label, confidence, hand_present)
        draw_ui(frame, builder.stabilizer.candidate, confidence, builder.stabilizer.progress, builder.text)
        cv2.imshow("ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == 8:
            builder.backspace()
        elif key == ord("c"):
            builder.clear()
        elif key == ord("t") and builder.text.strip():
            speak_text(builder.text)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

#-------------- Entry Point -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--task_path", default="hand_landmarker.task")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--hold_frames", type=int, default=15)
    parser.add_argument("--confidence_threshold", type=float, default=0.75)
    args = parser.parse_args()
    main(args.model_dir, args.task_path, args.camera_index, args.hold_frames, args.confidence_threshold)