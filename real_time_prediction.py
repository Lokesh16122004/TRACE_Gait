import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# CONFIG (MUST MATCH TRAINING)
IMG_SIZE = 64
SEQ_LEN = 10
CHUNK_SIZE = 20

people = ["Person_2", "Person_3", "Person_Anurag", "Person_Hemant", "Person_Lokesh"]

# LOAD MODEL
model = tf.keras.models.load_model("gait_model.h5")

# INIT MEDIAPIPE
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# STORAGE
frames_buffer = []

# START CAMERA
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # SILHOUETTE EXTRACTION
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentor.process(rgb)

    mask = result.segmentation_mask
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    condition = mask > 0.5

    silhouette = np.zeros_like(frame)
    silhouette[condition] = (255,255,255)

    gray = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    frames_buffer.append(gray / 255.0)

    # KEEP BUFFER SIZE
    if len(frames_buffer) > CHUNK_SIZE:
        frames_buffer.pop(0)

    # PREDICTION
    if len(frames_buffer) == CHUNK_SIZE:

        gei = np.mean(frames_buffer, axis=0)

        sequence = [gei for _ in range(SEQ_LEN)]
        X = np.array(sequence).reshape(1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 1)

        pred = model.predict(X, verbose=0)
        person = people[np.argmax(pred)]
        confidence = np.max(pred)

        cv2.putText(frame, f"{person} ({confidence:.2f})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    # DISPLAY
    cv2.imshow("Frame", frame)
    cv2.imshow("Silhouette", gray)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()