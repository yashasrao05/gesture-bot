

import cv2
import numpy as np
import pickle
import sys
from collections import deque
from pathlib import Path
import tensorflow as tf
import mediapipe as mp

import warnings
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated"
)


class GestureInference:
    def __init__(
        self,
        model_path='D:/Projects/Gesture-bot/model/gesture_model.keras',
        scaler_path='D:/Projects/Gesture-bot/model/gesture_scaler.pkl',
        classes_path='D:/Projects/Gesture-bot/model/gesture_classes.pkl'
    ):
        """
        Load trained model and scaler
        """

        print("GESTURE INFERENCE ENGINE")

        
        # Check files exist
        for f in [model_path, scaler_path, classes_path]:
            if not Path(f).exists():
                print(f"\n Missing: {f}")
                print("Run train_model.py first")
                sys.exit(1)
        
        print("\nLoading model...")
        self.model = tf.keras.models.load_model(model_path)
        
        print("Loading scaler...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("Loading classes...")
        with open(classes_path, 'rb') as f:
            self.gesture_classes = pickle.load(f)
        
        self.class_to_idx = {g: i for i, g in enumerate(self.gesture_classes)}
        self.idx_to_class = {i: g for g, i in self.class_to_idx.items()}
        
        # MediaPipe setup
        print("\nInitializing MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Smoothing: keep last N predictions
        self.smoothing_window = 5
        self.gesture_history = deque(maxlen=self.smoothing_window)
        self.confidence_history = deque(maxlen=self.smoothing_window)
        
        # Confidence threshold
        self.confidence_threshold = 0.6
        
        print("Gesture inference engine ready\n")
    
    def extract_landmarks(self, frame):
        """
        Extract normalized hand landmarks from frame
        Returns:
            features (63,), landmarks (21,3) or (None, None)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, None

        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        # Normalize: wrist as origin
        wrist = landmarks[0]
        normalized = landmarks - wrist

        # Scale by wrist → middle fingertip distance
        scale = np.linalg.norm(normalized[12])
        if scale < 1e-6:
            scale = 1.0

        normalized /= scale
        features = normalized.flatten()  # 63 features

        return features, landmarks
    
    def predict_gesture(self, features):
        """
        Predict gesture from features
        Returns: gesture_name, confidence
        """
        if features is None:
            return 'UNKNOWN', 0.0
        
        # Standardize features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        predictions = self.model.predict(features_scaled, verbose=0)[0]
        confidence = float(np.max(predictions))
        class_idx = np.argmax(predictions)
        gesture = self.idx_to_class[class_idx]
        
        return gesture, confidence
    
    def smooth_gesture(self, gesture, confidence):
        """
        Apply temporal smoothing to reduce flickering
        - Keep history of last N predictions
        - Return most common gesture
        - Return average confidence
        """
        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)
        
        if len(self.gesture_history) == 0:
            return 'UNKNOWN', 0.0
        
        # Most common gesture
        from collections import Counter
        counts = Counter(self.gesture_history)
        smoothed_gesture = counts.most_common(1)[0][0]
        
        # Average confidence of smoothed gesture
        smoothed_confidence = np.mean([
            c for g, c in zip(self.gesture_history, self.confidence_history)
            if g == smoothed_gesture
        ])
        
        return smoothed_gesture, smoothed_confidence
    
    def run_inference(self, visualize=True):
        """
        Main inference loop
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting inference... (Press Q to quit)")
        print("-" * 70)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Extract features
                features, landmarks = self.extract_landmarks(frame)
                
                # Predict
                raw_gesture, raw_confidence = self.predict_gesture(features)
                
                # Smooth
                smoothed_gesture, smoothed_confidence = self.smooth_gesture(raw_gesture, raw_confidence)
                
                # Apply confidence threshold
                if smoothed_confidence < self.confidence_threshold:
                    final_gesture = 'NONE'
                    final_confidence = 0.0
                else:
                    final_gesture = smoothed_gesture
                    final_confidence = smoothed_confidence
                
                if visualize:
                    # Draw on frame
                    cv2.putText(frame, "Gesture Recognition", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if landmarks is not None:
                        # Draw landmarks
                        for lm in landmarks:
                            x = int(lm[0] * w)
                            y = int(lm[1] * h)
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                        
                        # Draw gesture info
                        cv2.putText(frame, f"Gesture: {final_gesture}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        cv2.putText(frame, f"Confidence: {final_confidence:.2f}", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, f"(Raw: {raw_gesture} {raw_confidence:.2f})", (10, 140),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    else:
                        cv2.putText(frame, "No hand detected", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imshow('Gesture Inference', frame)
                
                # Print result
                if frame_count % 5 == 0:  # Print every 5 frames
                    print(f"Frame {frame_count:4d}: {final_gesture:8s} " +
                          f"({final_confidence:.2f}) | " +
                          f"Raw: {raw_gesture:8s} ({raw_confidence:.2f})")
                
                frame_count += 1
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting inference...")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")


if __name__ == '__main__':
    inference = GestureInference()
    inference.run_inference(visualize=True)