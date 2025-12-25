import asyncio
import websockets
import json
import cv2
import numpy as np
import pickle
import sys
import threading
from collections import deque, Counter
from pathlib import Path
import tensorflow as tf
import mediapipe as mp

from gesture_controller import GestureController


class GestureInferenceServer:
    def __init__(
        self,
        model_path='D:/Projects/Gesture-bot/model/gesture_model.keras',
        scaler_path='D:/Projects/Gesture-bot/model/gesture_scaler.pkl',
        classes_path='D:/Projects/Gesture-bot/model/gesture_classes.pkl',
        host='0.0.0.0',
        port=8082
    ):
        self.host = host
        self.port = port
        self.clients = set()
        self.is_running = False


        print(f" GESTURE INFERENCE + UI SERVER ({self.host}:{self.port})")

        # Load AI assets
        for f in [model_path, scaler_path, classes_path]:
            if not Path(f).exists():
                print(f" Missing: {f}")
                sys.exit(1)

        self.model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(classes_path, 'rb') as f:
            self.gesture_classes = pickle.load(f)

        self.idx_to_class = {i: g for i, g in enumerate(self.gesture_classes)}

        print(f" Loaded classes: {self.gesture_classes}")

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Gesture logic
        self.controller = GestureController()
        self.smoothing_window = 5
        self.gesture_history = deque(maxlen=self.smoothing_window)
        self.confidence_history = deque(maxlen=self.smoothing_window)
        self.confidence_threshold = 0.6


    # AI PIPELINE


    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark],
            dtype=np.float32
        )

        wrist = landmarks[0]
        normalized = landmarks - wrist
        scale = np.linalg.norm(normalized[12]) or 1.0
        normalized /= scale

        return normalized.flatten(), landmarks

    def predict(self, features):
        if features is None:
            return "NONE", 0.0

        scaled = self.scaler.transform([features])
        preds = self.model.predict(scaled, verbose=0)[0]

        idx = np.argmax(preds)
        return self.idx_to_class[idx], float(preds[idx])

    def smooth(self, gesture, confidence):
        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)

        counts = Counter(self.gesture_history)
        g = counts.most_common(1)[0][0]
        c = np.mean([c for gg, c in zip(self.gesture_history, self.confidence_history) if gg == g])
        return g, c



    def camera_worker(self, loop):
        cap = cv2.VideoCapture(0)
        print("Camera started")

        frame_id = 0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                features, landmarks = self.extract_landmarks(frame)

                raw_g, raw_c = self.predict(features)
                g, c = self.smooth(raw_g, raw_c)

                if c < self.confidence_threshold:
                    g, c = "STOP", 0.0

                motor_cmd = self.controller.gesture_to_command(g, c)

                msg = {
                    "type": "gesture_detected",
                    "gesture": motor_cmd.gesture,
                    "confidence": motor_cmd.confidence,
                    "left_motor": motor_cmd.left,
                    "right_motor": motor_cmd.right
                }

                # UI
                cv2.putText(frame, "Gesture Control", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"{g}  ({c:.2f})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if landmarks is not None:
                    h, w, _ = frame.shape
                    for lm in landmarks:
                        cv2.circle(frame,
                                   (int(lm[0] * w), int(lm[1] * h)),
                                   3, (0, 255, 0), -1)

                cv2.imshow("Gesture Inference", frame)

                # Broadcast
                if self.clients and frame_id % 5 == 0:
                    asyncio.run_coroutine_threadsafe(self.broadcast(msg), loop)

                if frame_id % 20 == 0:
                    print(f"[LIVE] {g:6s} | L:{motor_cmd.left:4d} R:{motor_cmd.right:4d}")

                frame_id += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False


    # WEBSOCKET


    async def handle_client(self, ws):
        print(f"[WS] Client connected: {ws.remote_address}")
        self.clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    async def broadcast(self, message):
        data = json.dumps(message)
        dead = set()
        for c in self.clients:
            try:
                await c.send(data)
            except:
                dead.add(c)
        self.clients -= dead

    async def run(self):
        self.is_running = True
        loop = asyncio.get_running_loop()

        threading.Thread(target=self.camera_worker, args=(loop,), daemon=True).start()

        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"WebSocket running on {self.port}")
            await asyncio.Future()

    def start(self):
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            self.is_running = False
            print("\nShutting down...")


if __name__ == "__main__":
    GestureInferenceServer().start()
