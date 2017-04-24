from skimage import io
from random import randint
from numpy.fft import fft2

def importImg (n):
    return io.imread('./img/{:04d}.png'.format(n), as_grey=True)

def extractSubImg (img, x, y, size):
    x = randint(0, 256 - size)
    y = randint(0, 256 - size)
    return img[x:(x+size), y:(y+size)]

def extractRandSubImg (img, size):
    x = randint(0, 256 - size)
    y = randint(0, 256 - size)
    return extractSubImg(img, x, y, size)

numpy.fft.fft2

print importImg(1)
print extractRandSubImg(importImg(1), 64)
