from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

labels_f = pd.read_csv('datasets/label.csv', delimiter=';', header=None)

categories = np.array(labels_f.iloc[:, 0])
# descriptions = np.array(labels_f.iloc[:, 1])

save = False
stream = True
conf = 0.2
target_size = (128, 128)
# model_path = 'models/vgg16_model.h5'
# model_path = 'models/mobilenet_model.h5'
model_path = 'models/inceptionV3_model.h5'

# Load model YOLOv8 custom
yolo = YOLO('models/yolov8-custom.pt')
# Load model du doan
model = keras.models.load_model(model_path)

# Detect tren video
cap = cv2.VideoCapture('test_vid.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None,fx=0.5,fy=0.5)
    # Phat hien bien bao giao thong
    detections = yolo(frame, save=save, stream=stream, conf=conf, imgsz=320)
    # Ve bounding box
    for detect in detections:
        for bbox in detect.boxes.xyxy:
            # Xac dinh toa do bounding box
            x1, y1, x2, y2 = np.array(bbox)
            # Du doan ma bien bao
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            roi = cv2.resize(roi, target_size)
            img_batch = np.expand_dims(roi, axis=0)
            img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
            predict = model.predict(img_preprocessed)
            labels_idx = np.argmax(predict, axis=1)
            label = categories[labels_idx[0]]
            if label == 'EMPTY':
                continue
            # Ve bounding box voi ma bien bien bao
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(label, font, 0.8, 2)

            bg_text_x1 = int(x1)
            bg_text_y1 = int(y1) - text_size[1] - 12
            bg_text_x2 = int(x1) + text_size[0] + 10
            bg_text_y2 = int(y1) - 5

            cv2.rectangle(frame, (bg_text_x1, bg_text_y1), (bg_text_x2, bg_text_y2), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Traffic sign detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()