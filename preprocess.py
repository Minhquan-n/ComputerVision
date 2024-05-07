from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import random
from sklearn.preprocessing import LabelBinarizer

# Ham luu tat ca anh da doc vao file datas.data
def save_data(target_path, image, label):
    
    pixels = np.array(image)
    labels = np.array(label)#.reshape(-1,1)

    encoder = LabelBinarizer()
    label_vec = encoder.fit_transform(labels)
    print(label_vec)

    file = open(target_path, 'wb')
    # dump information to that file
    pickle.dump((pixels,label_vec), file)
    # close the file
    file.close()
    print('Saved all image')

# Ham doc anh tu file datas.data
def load_data(path):
    file = open(path, 'rb')

    # Load thong tin trong file .data
    (pixels, label_ids) = pickle.load(file)

    file.close()

    print(pixels.shape)
    print(label_ids.shape)

    return pixels, label_ids

# Tai len mo hinh YOLOv8 da huan luyen tren du lieu custom
print('Load model YOLOv8 custom...')
yolo = YOLO('models/yolov8-custom.pt')
print('YOLOv8 ready.')

image_paths = []

non_detect = ['EMPTY', 'I.423b', 'I.425', 'I.409', 'I.414', 'I.434a', 'I.435', 'I.439']

pixels = []
labels = []

target_size = (128, 128)
# Doc tat ca file anh va nhan
base_path = 'datasets/data'
for folder in os.listdir(base_path):
    for file in os.listdir(base_path+'/'+folder):
        image_paths.append([base_path+'/'+folder+'/'+file, folder])

random.shuffle(image_paths)

# Duyet qua tung anh, detect bien bao giao thong trong anh, cat anh tai khu vuc co bien bao giao thong va luu lai vao pixels
for image in image_paths:
    img = cv2.imread(image[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if image[1] in non_detect:
        print(image[1])
        empty = cv2.resize(img, target_size)
        pixels.append(empty)
        labels.append(image[1])
        continue
    
    result = yolo(img, save=False, stream=True, conf=0.5, imgsz=320)
    for res in result:
        if res.boxes.xyxy.shape[0] < 1:
            nodetect = cv2.resize(img, target_size)
            pixels.append(nodetect)
            labels.append(image[1])
            continue
        for i, box in enumerate(res.boxes.xyxy):
            x1, y1, x2, y2 = np.array(box)

            roi = img[int(y1):int(y2), int(x1):int(x2)]
            roi = cv2.resize(roi, target_size)

            pixels.append(roi)
            labels.append(image[1])

print('Save image to data file...')
save_data('datasets/datas.data', pixels, labels)

# Tai anh len tu file de kiem tra
print('Load image from data file...')
data, label_ids = load_data('datasets/datas.data')

labels_f = pd.read_csv('datasets/label.csv', delimiter=';', header=None)
categories = np.array(labels_f.iloc[:, 0])

plt.subplots(3, 4)
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[np.argmax(label_ids[i])])
plt.show()
            