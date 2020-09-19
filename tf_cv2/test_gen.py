import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

size = (768,1024,1)
n = 100

def main():
    model = keras.models.load_model("test_256.h5")

    avg = 0
    for _ in range(n):
        img, pos = gen()
        x = np.array([img])
        y = np.array([[pos[0]],[pos[1]],[pos[2]]]).transpose()

        #print()
        #print("Predict :", model.predict(x)[0])
        #print("Expect  : [",pos[0],",",pos[1],",",pos[2],"]")
        #print()
        lost = model.test_on_batch(x,y)
        #print("Lost    :", lost)
        avg = avg + lost
    avg = avg / n
    print()
    print("Average :", avg)

def gen():
    img = np.zeros(size, np.uint8) # GrayScale Empty Image
    #img = cv2.imread(f, cv2.IMREAD_GRAYSCALE) # Load Image in GrayScale
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
