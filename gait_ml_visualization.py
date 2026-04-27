import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# USER INPUT (VERY IMPORTANT)
video_path = "test_video.mp4"   
label = "normal"


# INITIALIZE MEDIA PIPE
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(video_path)

left_knee_series = []
right_knee_series = []
stride_series = []

# PROCESS VIDEO
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

        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Processing", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# GAIT CYCLE DETECTION
left_knee_array = np.array(left_knee_series)
peaks, _ = find_peaks(left_knee_array, height=140, distance=10)

# FEATURE EXTRACTION
features = {
    "mean_knee": np.mean(left_knee_array),
    "std_knee": np.std(left_knee_array),
    "mean_stride": np.mean(stride_series),
    "symmetry": np.mean(np.abs(np.array(left_knee_series) - np.array(right_knee_series))),
    "cadence": len(peaks) / (len(left_knee_array) / 30),
    "label": label
}

print("\nExtracted Features:")
print(features)

# SAVE TO DATASET
csv_file = "gait_dataset.csv"

try:
    df = pd.read_csv(csv_file)
except:
    df = pd.DataFrame()

df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
df.to_csv(csv_file, index=False)

print("\nSaved to dataset.")

# TRAIN ML MODEL (only if enough data)
if len(df) > 5:  # minimum samples
    print("\nTraining ML Model...")

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
    print("\nNot enough data to train model yet.")