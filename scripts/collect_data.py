"""

- Captures hand landmarks using MediaPipe
- Records gesture labels (UP, DOWN, LEFT, RIGHT, STOP, NONE)
- Saves to CSV for training


"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

class GestureDataCollector:
    def __init__(self, output_file='D:/Projects/Gesture-bot/data-files/gesture_data.csv'):
        """
        Initialize data collector
        """
        print("Initializing MediaPipe Hands...")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.output_file = output_file
        self.gesture_map = {
            'u': 'UP',
            'd': 'DOWN',
            'l': 'LEFT',
            'r': 'RIGHT',
            's': 'STOP',
            'n': 'NONE'
        }
        
        # Create CSV file if not exists
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header: 21 landmarks * 3 (x, y, z) + gesture + confidence + timestamp
                header = []
                for i in range(21):
                    header.extend([f'x{i}', f'y{i}', f'z{i}'])
                header.extend(['gesture', 'timestamp'])
                writer.writerow(header)
        
        self.sample_count = {g: 0 for g in self.gesture_map.values()}
        print(f"Data collector ready. Output: {output_file}\n")
    
    def extract_landmarks(self, frame):
       """
       Extract hand landmarks from frame
       Returns: (21, 3) numpy array or None
       """
       h, w, _ = frame.shape
       rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       results = self.hands.process(rgb_frame)
    
       if not results.multi_hand_landmarks:
           return None
    
       landmarks = []
       for lm in results.multi_hand_landmarks[0].landmark:
           landmarks.append([lm.x, lm.y, lm.z])
    
       return np.array(landmarks)

    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks by translating to wrist (landmark 0) as origin
        This makes gestures invariant to hand position
        """
        if landmarks is None:
            return None
        
        wrist = landmarks[0]
        normalized = landmarks - wrist
        
        # Scale by distance from wrist to middle finger tip
        scale = np.linalg.norm(normalized[12])
        if scale < 1e-6:
            scale = 1.0
        
        normalized = normalized / scale
        
        return normalized.flatten()  # Flatten to 63 features (21*3)
    
    def record_sample(self, landmarks, gesture):
        """
        Save landmarks + gesture to CSV
        """
        if landmarks is None:
            return False
        
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = landmarks.tolist()
            row.extend([gesture, datetime.now().isoformat()])
            writer.writerow(row)
        
        self.sample_count[gesture] += 1
        return True
    
    def run_collection(self):
        """
        Main loop for data collection
        """

        print("GESTURE DATA COLLECTION")

        print("\nInstructions:")
        print("  U - Record UP gesture")
        print("  D - Record DOWN gesture")
        print("  L - Record LEFT gesture")
        print("  R - Record RIGHT gesture")
        print("  S - Record STOP gesture (open palm)")
        print("  N - Record NONE gesture (neutral/rest)")
        print("  Q - Quit collection")
        print("\nCapture 100-200 samples per gesture for good accuracy")
        print("Vary distance, hand size, and lighting")
        print("\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        last_recorded_gesture = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                normalized = self.normalize_landmarks(landmarks)
                
                # Draw info on frame
                cv2.putText(frame, "Hand Gesture Data Collection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if landmarks is not None:
                    # Draw hand landmarks
                    for i, lm in enumerate(landmarks):
                        x = int(lm[0] * w)
                        y = int(lm[1] * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    
                    cv2.putText(frame, "✓ Hand detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if last_recorded_gesture:
                        cv2.putText(frame, f"Last: {last_recorded_gesture}", (10, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "✗ No hand detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Show sample counts
                y_pos = 150
                for gesture, count in self.sample_count.items():
                    cv2.putText(frame, f"{gesture}: {count}", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_pos += 25
                
                cv2.imshow('Gesture Data Collection', frame)
                frame_count += 1
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n Quitting collection...")
                    break
                
                elif key in [ord(k) for k in self.gesture_map.keys()]:
                    gesture = self.gesture_map[chr(key)]
                    
                    if landmarks is not None:
                        if self.record_sample(normalized, gesture):
                            last_recorded_gesture = gesture
                            print(f"Recorded {gesture} (total: {self.sample_count[gesture]})")
                    else:
                        print(f"No hand detected - skipping {gesture}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_summary()
    
    def print_summary(self):
        """
        Print collection summary
        """

        print("COLLECTION SUMMARY \n \n")

        total = sum(self.sample_count.values())
        print(f"Total samples: {total}\n")
        
        for gesture, count in self.sample_count.items():
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * (count // 10)
            print(f"  {gesture:6s}: {count:3d} samples ({pct:5.1f}%) {bar}")
        
        print(f"\n Data saved to: {self.output_file}")
        print("Ready for training!")


if __name__ == '__main__':
    import sys
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'D:/Projects/Gesture-bot/data-files/gesture_data.csv'
    
    collector = GestureDataCollector(output_file)
    collector.run_collection()