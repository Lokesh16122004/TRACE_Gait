import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

VIDEO_ROOT = "Dataset_Gait/Videos"
DATASET_ROOT = "Dataset_Gait/Silhouettes"

IMG_SIZE = 64
SEQ_LEN = 10
CHUNK_SIZE = 20  # frames per GEI

# INIT MEDIAPIPE
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# STEP 1: EXTRACT SILHOUETTES
def extract_silhouettes():
    for person in os.listdir(VIDEO_ROOT):

        person_folder = os.path.join(VIDEO_ROOT, person)
        save_path = os.path.join(DATASET_ROOT, person)

        os.makedirs(save_path, exist_ok=True)

        for video_file in os.listdir(person_folder):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(person_folder, video_file)
            print(f"Processing: {video_path}")

            cap = cv2.VideoCapture(video_path)
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = segmentor.process(rgb)

                mask = result.segmentation_mask
                mask = cv2.GaussianBlur(mask, (7,7), 0)

                condition = mask > 0.5

                silhouette = np.zeros_like(frame)
                silhouette[condition] = (255,255,255)

                gray = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                filename = f"{video_file}_{frame_id}.png"
                cv2.imwrite(os.path.join(save_path, filename), gray)

                frame_id += 1

            cap.release()

# STEP 2: BUILD DATASET (GEI)
def build_dataset():
    X = []
    y = []
    people = os.listdir(DATASET_ROOT)

    for idx, person in enumerate(people):
        folder = os.path.join(DATASET_ROOT, person)

        frames = sorted(os.listdir(folder))

        for i in range(0, len(frames) - CHUNK_SIZE, CHUNK_SIZE):

            chunk = frames[i:i+CHUNK_SIZE]

            images = []
            for file in chunk:
                img = cv2.imread(os.path.join(folder, file), 0)
                images.append(img / 255.0)

            gei = np.mean(images, axis=0)

            sequence = [gei for _ in range(SEQ_LEN)]

            X.append(sequence)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    return X, y, people

# STEP 3: BUILD MODEL
def build_model(num_classes):
    model = models.Sequential()

    model.add(layers.TimeDistributed(
        layers.Conv2D(32, (3,3), activation='relu'),
        input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 1)
    ))

    model.add(layers.TimeDistributed(layers.MaxPooling2D(2,2)))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(2,2)))
    model.add(layers.TimeDistributed(layers.Flatten()))

    model.add(layers.LSTM(64))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# STEP 4: TRAIN
def train_model():
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    print("\nChecking silhouettes...")

    has_data = False

    if os.path.exists(DATASET_ROOT):
        for person in os.listdir(DATASET_ROOT):
            person_path = os.path.join(DATASET_ROOT, person)
            if os.path.isdir(person_path) and len(os.listdir(person_path)) > 0:
                has_data = True
                break

    if not has_data:
        print("\nExtracting silhouettes...")
        extract_silhouettes()
    else:
        print("\nSilhouettes already exist. Skipping extraction.")

    print("\nBuilding dataset...")
    X, y, people = build_dataset()

    print(f"Dataset size: {len(X)} samples")

    if len(X) < 5:
        print("Not enough data. Add more videos.")
        return None, None

    # PREPROCESSING
    X = X.reshape(len(X), SEQ_LEN, IMG_SIZE, IMG_SIZE, 1)
    X = X.astype("float32") / 255.0

    # Shuffle dataset
    X, y = shuffle(X, y, random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MODEL
    model = build_model(len(people))

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=70,          # you can increase later
        batch_size=8
    )

    # SAVE MODEL
    model.save("gait_model.h5")

    # VISUALIZATION: ACCURACY & LOSS
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # CONFUSION MATRIX
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=people)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # SAMPLE PREDICTIONS
    print("\nSample Predictions:")
    for i in range(min(5, len(X_test))):
        plt.imshow(X_test[i][0], cmap='gray')
        plt.title(
            f"True: {people[y_test[i]]} | Pred: {people[np.argmax(y_pred[i])]}"
        )
        plt.axis('off')
        plt.show()

    return model, people

# STEP 5: PREDICT
def predict(model, people, X):
    pred = model.predict(X)
    for i, p in enumerate(pred):
        print(f"Sample {i} → Predicted: {people[np.argmax(p)]}")

# RUN
model, people = train_model()

if model:
    X, _, _ = build_dataset()
    X = X.reshape(len(X), SEQ_LEN, IMG_SIZE, IMG_SIZE, 1)

    print("\nPredictions:")
    predict(model, people, X)