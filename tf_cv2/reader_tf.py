# https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=zh-tw

import numpy as np
import cv2
from progressbar import *
import argparse
import time, os

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

p = argparse.ArgumentParser(description="A script that trains a model and save it as *.h5")
p.add_argument('-r', '--repeats', required=False, help='Set how much times the dataset repeats. (Default 16)', dest="r", type=int, default=16)
p.add_argument('-f', '--file', required=True, help='Specify the training file (must be a valid TFRecords file)', dest="f")
args = vars(p.parse_args())
if(not is_int(args["r"])):
    p.error(str(args["r"])+" is not a int")
#if(args["f"] == None):
#    p.error("File not specified. Add -f <file> to continue.")
if(not os.path.isfile(str(args["f"]))):
    p.error("File not found : "+str(args["f"]))

import tensorflow as tf
from tensorflow import keras

t = time.strftime("%j%H%M%S", time.localtime()) # this is a string of numbers

#raw_dataset = tf.data.TFRecordDataset("tfrecords/249030600.tfrecords")
#raw_dataset = tf.data.TFRecordDataset("tfrecords\\249173041.tfrecords")
raw_dataset = tf.data.TFRecordDataset(str(args["f"]))

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

if(int(args["r"]) < 64):
    dataset = dataset.repeat(int(args["r"]))
    dataset = dataset.shuffle(16384) # SHUFFLE_BUFFER
    dataset = dataset.batch(64) # BATCH_SIZE
    total = len([0 for _ in dataset])
else:
    total = 0
    for _ in dataset:
        total = total + 1
    
    dataset = dataset.repeat(int(args["r"]))
    dataset = dataset.shuffle(16384) # SHUFFLE_BUFFER
    dataset = dataset.batch(64) # BATCH_SIZE

    total = np.ceil(total * int(args["r"]) / 64)

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
    keras.layers.Dense(32),
    keras.layers.Dense(32),
    keras.layers.Dense(3 ) # final output
])

model.summary()

# Train
optimizer = keras.optimizers.RMSprop(0.001)
loss_function = keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss_function, metrics=[])

#batchsize = 256
#for i in xrange(0, len(dataset), batchsize):
#    batch = dataset[i:i+batchsize]
widgets = [
    FormatLabel("[%(value)"+str(int(np.log10(total))+1)+"d"), "/",
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
    #y_train = [batch["x"].numpy()/2000+0.5,batch["y"].numpy()/1000,batch["z"].numpy()/1000]
    y_train = [batch["x"].numpy(),batch["y"].numpy(),batch["z"].numpy()]
    y_train = np.array(y_train).transpose()
    model.train_on_batch(x_train, y_train)
    i = i + 1
    pbar.update(i)

pbar.finish()
"""
model.fit(x_train, y_train, epochs=10, validation_split=0.05)
"""
model.save("test_"+str(args["r"])+".h5")

# Test
#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])