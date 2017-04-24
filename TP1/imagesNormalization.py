import skimage.io
import skimage.color
import skimage.transform
import os
import numpy as np

inputDirectory = "../raw/"
outputDirectory = "../img/"


def normalizeAndSaveImages( inputDirectory, outputDirectory):
    """ Function that loads, normalize and save all the images  in a directory into an other directory
    Args:
        inputDirectory (str) the directory where the images are stored. The absolute path should be given
        outputDirectory (str) the directory where the images are going to be saved. The absolute path should be given
    """
    listOfFiles = os.listdir(inputDirectory)
    for i in range(len(listOfFiles)):
        inputFileName = inputDirectory + listOfFiles[i]
        print inputFileName
        img = skimage.io.imread(inputFileName)
        img = skimage.transform.resize(img,[256,256]);
        img = skimage.color.rgb2gray(img);
        outputFileName = outputDirectory+"image"+np.str(i)+".png"
        newimg =(((img-img.min())/img.max())*256).astype("uint8") #this is to normalize the values to 8bit, necessary for png images
        skimage.io.imsave(outputFileName,newimg)


normalizeAndSaveImages(inputDirectory,outputDirectory)
