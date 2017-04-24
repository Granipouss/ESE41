from skimage import io, transform, color
from os import listdir
from os.path import isfile, join

i = 1
def convert (filename):
    global i
    img = io.imread(filename)
    img = transform.resize(img, [256, 256])
    img = color.rgb2grey(img)
    io.imsave('./img/{:04d}.png'.format(i), img)
    i += 1

rawDir = './raw_img'
filenames = [f for f in listdir(rawDir) if isfile(join(rawDir, f))]

for filename in filenames:
    convert('./raw_img/' + filename)
