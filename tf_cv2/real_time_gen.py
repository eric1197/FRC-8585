import time, os
import cv2
import numpy as np
import argparse
from progressbar import *

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# n = 10 # size of dataset
# size = (768, 1024) # size of image
p = argparse.ArgumentParser(description="A script that generates TFRecords as training resources.")
p.add_argument('-n', '--size', required=False, help='Set the size of dataset.', dest="n", type=int, default=100)
p.add_argument('-s', '--screen', required=False, help='Set the size of screen. Ex. 1024x768', dest="s", default="1024x768")
p.add_argument('-g', '--gui',action='store_true',help='Enable the output screen. XD (Slow down training a lot)')
args = vars(p.parse_args())
s = args["s"].split("x")
if (len(s) != 2) or not is_int(s[0]) or not is_int(s[1]):
    p.error("Screen size not valid ("+args["s"]+")")

n = args["n"]
size = (int(s[1]),int(s[0]))

import tensorflow as tf
from tensorflow import keras

t = time.strftime("%j%H%M%S", time.localtime()) # this is a string of numbers
path = os.path.abspath(os.getcwd())
filename = os.path.join(path, "tfrecords", t+".tfrecords")
np.random.seed(int(t))

def main():

    # Setup
    model = keras.models.Sequential([
        keras.Input(shape=size),
        keras.layers.Flatten(),
        keras.layers.Dense(32),
        keras.layers.Dense(32),
        keras.layers.Dense(3 ) # final output
    ])
    
    model.summary()

    optimizer = keras.optimizers.RMSprop(0.001)
    loss_function = keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[])

    widgets = [
        FormatLabel("[%(value)"+str(int(np.log10(n))+1)+"d"), "/",
        FormatLabel("%(max)d"), "]",
        Percentage(), " ",
        Bar(marker="=",left="[",right="]",fill="-"), " ",
        FormatLabel("Timer: %(elapsed)s"), " / ", ETA()
    ]
    pbar = ProgressBar(widgets=widgets, maxval=n).start()
    pbar.update(0)

    for i in np.arange(n):
        img, pos = gen()
        x = np.array([img])
        y = np.array([[pos[0]],[pos[1]],[pos[2]]]).transpose()
        if args["gui"]:
            cv2.imshow("image", img)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF is ord('q'):
                break
        model.train_on_batch(x, y)
        pbar.update(i)
    pbar.finish()

    img, pos = gen()
    x = np.array([img])
    y = np.array([[pos[0]],[pos[1]],[pos[2]]]).transpose()
    lost = model.test_on_batch(x, y)
    print()
    print()
    print("Lost :", lost)

    path = os.path.join(".","h5")
    if not os.path.isdir(path):
        os.makedirs(path)
    model.save(os.path.join(path,"test_"+t+".h5"))
    
    if args["gui"]:
        cv2.destroyAllWindows()

def gen():
    img = np.zeros(size, np.uint8) # GrayScale Empty Image
    img.fill(0) # Fill with black

    x = -760 + np.random.rand() * 800
    y = 1600/2 - np.random.rand() * 500
    z = 249 -60 - np.random.rand()*20
    vx = (np.random.rand() * 60 - 30) * np.pi / 180
    vy = (np.random.rand() * 60 - 30) * np.pi / 180

    x = int(x)
    y = int(y)
    z = int(z)

    coordinates = draw(x, y, z, size, hex_size=44, vision_offset=(vx, vy))
    for j in range(3):
        cv2.line(img, coordinates[j], coordinates[j+1], 255, 7)

    return img,[x,y,z]

def convert_polar(x, y, z):
    r = np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))
    if x != 0:
        a = np.arctan(y/x)
    else:
        a = np.pi / 2
    # Handle arctan Domain
    if x < 0:
        a = a + np.pi
    b = np.arcsin(z/r)
    return r, a, b

sqrt3 = np.sqrt(3)
offset = np.array([[-1,0,0],[-0.5,0,-sqrt3/2],[0.5,0,-sqrt3/2],[1,0,0]])

# Calculate the 4 cordinates to draw on screen
# The 4 coordinates are the buttom 4 point of the hexagon
# `hex_size` is the length of one side
def draw(x, y, z, size, hex_size=5, vision_offset=(0,0)):
    p = np.full((4,3), [x, y, z]) + offset * hex_size # the 4 coordinates
    cr, ca, cb = convert_polar(x, y, z)
    # vision (with offset)
    ca = ca+vision_offset[0]
    cb = cb+vision_offset[1]
    
    q = []
    for i in p:
        r, a, b = convert_polar(i[0], i[1], i[2])
        r -= cr
        a -= ca
        b -= cb
        size_min = min(size[0], size[1])
        m = size[1] / 2 - size_min / 2 * a
        n = size[0] / 2 - size_min / 2 * b
        q.append((int(m), int(n)))
        #print(i[0],a/np.pi*180)
    return q

if __name__ == '__main__':
    main()
