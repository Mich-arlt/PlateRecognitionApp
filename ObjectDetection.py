import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as xet
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model


def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find("filename").text
    filename_image = os.path.join("./images", filename_image)
    return filename_image


# Załadowanie danych i pobranie naz zdjęć

df = pd.read_csv("labels.csv")
filename = df["filepath"][0]
image_path = list(df["filepath"].apply(getFilename))
file_path = image_path[0]

# weryfikacja danych

img = cv2.imread(file_path)

cv2.rectangle(img, (534, 320), (626, 344), (0, 0, 255), 1)
cv2.namedWindow("example", cv2.WINDOW_NORMAL)
cv2.imshow("example", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Przygotowanie do przetwarzania danych

labels = df.iloc[:, 1:].values
data = []
output = []

for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h, w, d = img_arr.shape
    load_image = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    load_image_arr = tf.keras.preprocessing.image.img_to_array(load_image)
    norm_load_image_arr = load_image_arr / 255.0
    xmin, xmax, ymin, ymax = labels[ind]
    nxmin, nxmax = xmin / w, xmax / w
    nymin, nymax = ymin / h, ymax / h
    label_norm = (nxmin, nxmax, nymin, nymax)
    data.append(norm_load_image_arr)
    output.append(label_norm)

X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0
)

# trenowanie modelu

inception_resnet = InceptionResNetV2(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)
inception_resnet.trainable = False

headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500, activation="relu")(headmodel)
headmodel = Dense(250, activation="relu")(headmodel)
headmodel = Dense(4, activation="sigmoid")(headmodel)

model = Model(inputs=inception_resnet.input, outputs=headmodel)

# komplilowanie modelu

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()


tfb = TensorBoard("object_detection")
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=10,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tfb],
)

# trenowanie modelu

tfb = TensorBoard("object_detection")
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=10,
    epochs=200,
    validation_data=(x_test, y_test),
    callbacks=[tfb],
    initial_epoch=101,
)
model.save("web_app/static/Models/object_detection.h5")
