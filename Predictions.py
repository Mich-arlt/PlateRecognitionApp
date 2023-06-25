import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

model = keras.models.load_model("./Models/object_detection.h5")
print("model loaded sucessfully")
path = "./images/tablice-rejestracyjne-Ford-Mustang.jpg"


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


image, coords = object_detection(path)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.show()
