import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

# Choose input: 0 = webcam OR "test_video.mp4"
cap = cv2.VideoCapture("D:\TRACE_Gait\Dataset\VID_20260421_234355.mp4")

step_count = 0
direction = None
start_time = time.time()
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentor.process(rgb)

    mask = result.segmentation_mask

    # Convert to binary silhouette
    condition = mask > 0.5
    silhouette = np.zeros_like(frame)
    silhouette[condition] = (255, 255, 255)

    cv2.imshow("Silhouette DL", silhouette)

    mask = cv2.GaussianBlur(mask, (7,7), 0)
    condition = mask > 0.5
    mask = fgbg.apply(frame)
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Silhouette", thresh)

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

        # STEP DETECTION (basic logic)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        if avg_knee_angle > 165:
            if direction == "down":
                step_count += 1
                direction = "up"
        if avg_knee_angle < 130:
            direction = "down"

        # CADENCE
        elapsed_time = time.time() - start_time
        cadence = step_count / elapsed_time if elapsed_time > 0 else 0

        # DISPLAY
        cv2.putText(frame, f"Steps: {step_count}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Cadence: {cadence:.2f} steps/sec", (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.putText(frame, f"L Knee: {int(left_knee_angle)}", (30,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"R Knee: {int(right_knee_angle)}", (30,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Gait Analysis", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
