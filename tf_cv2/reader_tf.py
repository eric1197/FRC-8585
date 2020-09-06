# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=zh-tw

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from progressbar import *

#raw_dataset = tf.data.TFRecordDataset("tfrecords/249030600.tfrecords")
raw_dataset = tf.data.TFRecordDataset("tfrecords\\249173041.tfrecords")

# Create a description of the features.
feature_description = {
    'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'x': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'y': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'z': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

dataset = raw_dataset.map(_parse_function)

dataset = dataset.repeat(16)
dataset = dataset.shuffle(16384) # SHUFFLE_BUFFER
dataset = dataset.batch(8) # BATCH_SIZE

#for parsed_record in dataset.take(10):
#    print(repr(parsed_record)[:200])

# tf.image.decode_jpeg(img_raw)
"""
x_train = []
y_train = []
for record in dataset:
    x_train.append(tf.image.decode_png(record["image_raw"]).numpy().transpose()[0])
    y_train.append([record["x"].numpy(),record["y"].numpy(),record["z"].numpy()])

x_train = np.array(x_train)
y_train = np.array(y_train)
"""
for batch in dataset.take(1):
    input_shape = (batch["height"].numpy()[0], batch["width"].numpy()[0], 1)

# Setup
model = keras.models.Sequential([
    keras.Input(shape=input_shape),
#    keras.layers.Conv2D(256, 1, activation="relu"),
#    keras.layers.Conv2D(128, (1,3), activation="relu"),
#    keras.layers.MaxPooling2D(3, data_format='channels_last'),
#    keras.layers.Conv2D(64, 1, activation="relu"),
#    keras.layers.Conv2D(32, (1,2), activation="relu"),
#    keras.layers.MaxPooling2D(2, data_format='channels_last'),
#    keras.layers.Conv2D(16, 1, activation="relu"),
#    keras.layers.Conv2D(8, (1,2), activation="relu"),
#    keras.layers.MaxPooling2D(2, data_format='channels_last'),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3 , activation="softmax") # final output
])

model.summary()

# Train
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#batchsize = 256
#for i in xrange(0, len(dataset), batchsize):
#    batch = dataset[i:i+batchsize]
total = 20000
widgets = [
    FormatLabel("[Progress: %(value)"+str(int(np.log10(total))+1)+"d"), "/",
    FormatLabel("%(max)d"), "]",
    Percentage(), " ",
    Bar(marker="=",left="[",right="]",fill="-"), " ",
    FormatLabel("Timer: %(elapsed)s"), " / ", ETA()
]
pbar = ProgressBar(widgets=widgets, maxval=total).start()
i = 0
pbar.update(0)

for batch in dataset:
    x_train = []
    for img_raw in batch["image_raw"].numpy():
        img = np.fromstring(img_raw, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(input_shape)
        x_train.append(img)
    x_train = np.array(x_train)
    y_train = [batch["x"].numpy()/2000+0.5,batch["y"].numpy()/1000,batch["z"].numpy()/1000]
    y_train = np.array(y_train).transpose()
    if i % 3 == 0:
        model.test_on_batch(x_train, y_train)
    else:
        model.train_on_batch(x_train, y_train)
    i = i + 1
    pbar.update(i)

pbar.finish()
"""
model.fit(x_train, y_train, epochs=10, validation_split=0.05)
"""
model.save("test.h5")

# Test
#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])