import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pytesseract as pt
import easyocr
from RotateImage import read_edges

# w tej lokalizacji będzie model
model = keras.models.load_model("static/Models/object_detection.h5")
print("model loaded sucessfully")
path = "./images/000FVWV8MS1T41RT-C122-F4.jpg"
pt.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"


def object_detection(path, filename):
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
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return coords

def OCR(path, filename):
    img = np.array(tf.keras.preprocessing.image.load_img(path))
    coords = object_detection(path, filename)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_to_save  =cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./static/roi/{}'.format(filename), roi_to_save)
    try:
        roi_rotated_raw = read_edges(roi)
    except:
        roi_rotated_raw = roi
    # roi_rotated = np.clip(roi_rotated_raw * 1.5, 0, 255).astype(np.uint8)
    roi_gray = cv2.cvtColor(roi_rotated_raw, cv2.COLOR_BGR2GRAY)
    roi_rotated_adjusted = cv2.equalizeHist(roi_gray)
    _, roi_binary = cv2.threshold(roi_rotated_adjusted, 127, 255, cv2.THRESH_BINARY)
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(roi_binary, width_ths=0.7, text_threshold=0.7, detail=0)
    final_answear = ' '.join(results)
    return str(final_answear)