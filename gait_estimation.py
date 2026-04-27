import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Angle calculation
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

# Video input (0 for webcam OR use "test_video.mp4")
cap = cv2.VideoCapture("D:\TRACE_Gait\Dataset\VID_20260421_223800.mp4")

# Data storage
left_knee_series = []
right_knee_series = []
stride_series = []

start_time = time.time()

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

        # STRIDE LENGTH
        stride = abs(
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x -
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        )

        # STORE DATA
        left_knee_series.append(left_knee_angle)
        right_knee_series.append(right_knee_angle)
        stride_series.append(stride)

        # DISPLAY
        cv2.putText(frame, f"L Knee: {int(left_knee_angle)}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"R Knee: {int(right_knee_angle)}", (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Gait Analysis", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===============================
# 🔥 GAIT CYCLE DETECTION
# ===============================

left_knee_array = np.array(left_knee_series)

# Detect peaks (gait cycles)
peaks, _ = find_peaks(left_knee_array, height=140, distance=10)

print("Total Gait Cycles Detected:", len(peaks))

# ===============================
# 🔥 FEATURE EXTRACTION
# ===============================

features = {
    "mean_knee": np.mean(left_knee_array),
    "std_knee": np.std(left_knee_array),
    "mean_stride": np.mean(stride_series),
    "symmetry": np.mean(np.abs(np.array(left_knee_series) - np.array(right_knee_series))),
    "cadence": len(peaks) / (len(left_knee_array) / 30)  # assuming 30 FPS
}

print("\nExtracted Features:")
for k, v in features.items():
    print(f"{k}: {v:.2f}")

# ===============================
# 📊 VISUALIZATION
# ===============================

plt.plot(left_knee_array, label="Left Knee Angle")
plt.scatter(peaks, left_knee_array[peaks])
plt.title("Gait Cycle Detection")
plt.xlabel("Frames")
plt.ylabel("Angle")
plt.legend()
plt.show()