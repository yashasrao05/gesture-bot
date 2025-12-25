"""
- Trains MLP classifier on MediaPipe hand landmarks
- Automatically infers gesture classes from data
- Saves model + scaler + class list

"""

import csv
import sys
import pickle
import numpy as np
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class GestureModelTrainer:
    def __init__(self, data_file='D:/Projects/Gesture-bot/data-files/gesture_data.csv'):
        self.data_file = data_file

        self.X = None
        self.y = None

        self.gesture_classes = None
        self.class_to_idx = None
        self.idx_to_class = None

        self.scaler = None
        self.model = None


        print("GESTURE MODEL TRAINER")


 

    def load_data(self):
        print(f"\n Loading data from {self.data_file}...")

        if not Path(self.data_file).exists():
            print(f"File not found: {self.data_file}")
            sys.exit(1)

        X, y_labels = [], []

        with open(self.data_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                features = [float(v) for v in row[:63]]
                gesture = row[63]

                X.append(features)
                y_labels.append(gesture)

        # Infer classes dynamically
        self.gesture_classes = sorted(list(set(y_labels)))
        self.class_to_idx = {g: i for i, g in enumerate(self.gesture_classes)}
        self.idx_to_class = {i: g for g, i in self.class_to_idx.items()}

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array([self.class_to_idx[g] for g in y_labels], dtype=np.int64)

        print(f" Loaded {len(self.X)} samples")
        print(f" Detected classes: {self.gesture_classes}")
        print(f" Feature shape: {self.X.shape}")

        print("\nClass distribution:")
        for g in self.gesture_classes:
            count = np.sum(self.y == self.class_to_idx[g])
            pct = count / len(self.y) * 100
            print(f"  {g:6s}: {count:3d} ({pct:5.1f}%)")

        if len(self.X) < 100:
            print("\n Warning: Very small dataset")



    def preprocess(self):
        print("\n Standardizing features...")
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        print("Done")



    def build_model(self):
        print("\n Building model...")

        num_classes = len(self.gesture_classes)

        self.model = Sequential([
            Input(shape=(63,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()



    def train(self, epochs=50, batch_size=32, test_size=0.2):
        print(f"\n Training for {epochs} epochs...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y
        )

        print(f"Train samples: {len(X_train)}")
        print(f"Test samples:  {len(X_test)}")

        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )

        return X_test, y_test



    def evaluate(self, X_test, y_test):
        print("\n Evaluating model...")

        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n Test Accuracy: {acc * 100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix:")
        header = "     " + "  ".join([g[:3] for g in self.gesture_classes])
        print(header)
        for i, row in enumerate(cm):
            print(f"{self.gesture_classes[i]:5s}:", "  ".join(f"{v:3d}" for v in row))

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=self.gesture_classes,
            digits=3
        ))

        return acc



    def save_model(self):
        print("\n Saving...")

        self.model.save("gesture_model.keras")
        with open("gesture_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open("gesture_classes.pkl", "wb") as f:
            pickle.dump(self.gesture_classes, f)

        print("gesture_model.keras")
        print("gesture_scaler.pkl")
        print("gesture_classes.pkl")



    def run(self):
        self.load_data()
        self.preprocess()
        self.build_model()
        X_test, y_test = self.train()
        acc = self.evaluate(X_test, y_test)
        self.save_model()


        print(" TRAINING COMPLETE")

        print(f"Final accuracy: {acc * 100:.2f}%")
        print("Next step: real-time inference")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="D:/Projects/Gesture-bot/data-files/gesture_data.csv")
    args = parser.parse_args()

    trainer = GestureModelTrainer(args.data)
    trainer.run()
