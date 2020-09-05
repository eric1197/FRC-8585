# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=zh-tw

import tensorflow as tf
from tensorflow import keras
import numpy as np

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

#dataset = dataset.repeat()
#dataset = dataset.shuffle(16384) # SHUFFLE_BUFFER
#dataset = dataset.batch(256) # BATCH_SIZE

#for parsed_record in dataset.take(10):
#    print(repr(parsed_record)[:200])

# tf.image.decode_jpeg(img_raw)
x_train = []
y_train = []
for record in dataset:
    x_train.append(tf.image.decode_png(record["image_raw"]).numpy().transpose()[0])
    y_train.append([record["x"].numpy(),record["y"].numpy(),record["z"].numpy()])

x_train = np.array(x_train)
y_train = np.array(y_train)

for record in dataset.take(1):
    input_shape = (record["width"].numpy(),record["height"].numpy())

# Setup
model = keras.models.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax') # final output
])

model.summary()

# Train
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.05)

model.save("test.h5")

# Test
#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])