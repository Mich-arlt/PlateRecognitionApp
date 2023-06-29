import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras_ocr
from RotateImage import read_edges
import easyocr

model = keras.models.load_model("web_app/static/Models/object_detection.h5")
print("model loaded sucessfully")
path = "./images/354810231_572217221653650_8569489559658457423_n.jpg"


def object_detection(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    image_arr_224 = tf.keras.preprocessing.image.img_to_array(image1) / 255.0

    h, w, d = image.shape

    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    coords = keras.models.Model.predict(model, test_arr)
    print(coords)
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = np.int32(coords)
    print(coords)

    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

    return image, coords

def OCR(path):
    img = np.array(tf.keras.preprocessing.image.load_img(path))
    plt.imshow(img)
    plt.show()
    _, coords = object_detection(path)
    print(coords)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    print(roi)
    roi_rotated_raw = read_edges(roi)
    plt.imshow(roi_rotated_raw)
    plt.show()
    roi_gray = cv2.cvtColor(roi_rotated_raw, cv2.COLOR_BGR2GRAY)
    plt.imshow(roi_gray)
    plt.show()
    print(np.mean(roi_gray))
    if np.mean(roi_gray) < 100:
        roi_binary = np.clip(roi_gray * 1.5, 0, 255).astype(np.uint8)
        _, roi_binary = cv2.threshold(roi_binary, 80, 255, cv2.THRESH_BINARY)
    elif np.mean(roi_gray) > 180:
        _, roi_binary = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
    else:
        roi_binary = np.clip(roi_gray * 1.5, 0, 255).astype(np.uint8)
        _, roi_binary = cv2.threshold(roi_binary, 127, 255, cv2.THRESH_BINARY)
    plt.imshow(roi_gray)
    plt.show()
    plt.imshow(roi_binary)
    plt.show()
    reader = easyocr.Reader(['en'], gpu=False)
    text = reader.readtext(roi_binary, width_ths=0.7, text_threshold=0.5, detail=0, paragraph=0)
    return roi_rotated_raw, text

def OCR_keras(path):
    img = np.array(tf.keras.preprocessing.image.load_img(path))
    _, coords = object_detection(path)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_rotated = read_edges(roi)
    pipeline = keras_ocr.pipeline.Pipeline()
    text = pipeline.recognize([roi])
    return roi_rotated, text

roi, text = OCR(path)
# wycięta tablica
plt.imshow(roi)
plt.show()
# odczytany tekst
print(text)
