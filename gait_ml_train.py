import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# INIT
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

# DATASET PATH
data_path = "D:\TRACE_Gait\Dataset"   # your folder

all_features = []

# PROCESS MULTIPLE VIDEOS
for label in ["normal", "abnormal"]:
    folder = os.path.join(data_path, label)

    for file in os.listdir(folder):
        if file.endswith(".mp4"):

            video_path = os.path.join(folder, file)
            print(f"\nProcessing: {video_path}")

            cap = cv2.VideoCapture(video_path)

            left_knee_series = []
            right_knee_series = []
            stride_series = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    # LEFT LEG
                    l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

                    # RIGHT LEG
                    r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                    # STRIDE
                    stride = abs(
                        lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x -
                        lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
                    )

                    # STORE
                    left_knee_series.append(left_knee_angle)
                    right_knee_series.append(right_knee_angle)
                    stride_series.append(stride)

            cap.release()

            # SKIP BAD VIDEOS
            if len(left_knee_series) < 10:
                print("Skipped (not enough data)")
                continue

            # GAIT CYCLE DETECTION
            left_knee_array = np.array(left_knee_series)
            peaks, _ = find_peaks(left_knee_array, height=140, distance=10)

            # FEATURE EXTRACTION
            features = {
                "mean_knee": np.mean(left_knee_array),
                "std_knee": np.std(left_knee_array),
                "mean_stride": np.mean(stride_series),
                "symmetry": np.mean(np.abs(
                    np.array(left_knee_series) - np.array(right_knee_series)
                )),
                "cadence": len(peaks) / (len(left_knee_array) / 30),
                "label": label
            }

            all_features.append(features)

# SAVE DATASET
df = pd.DataFrame(all_features)
df.to_csv("gait_dataset.csv", index=False)

print("\nDataset created:")
print(df.head())

# TRAIN MODEL
if len(df) > 5:
    print("\nTraining ML model...")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

else:
    print("\nNot enough data to train model.")