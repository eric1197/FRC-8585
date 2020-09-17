import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

input_shape = (768,1024)

img = cv2.imread("test_-377_480_175.png",0).reshape(input_shape)

x = np.array([img])

#y = np.array([[0.3115],[0.48],[0.175]]).transpose()
y = np.array([[-377],[480],[175]]).transpose()

model = keras.models.load_model("test.h5")

print()
print("Predict :", model.predict(x)[0])
print("Expect  : [-377, 480, 175]")
print()
lost = model.test_on_batch(x,y)
print("Lost    :", lost)