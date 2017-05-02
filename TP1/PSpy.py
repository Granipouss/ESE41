import numpy as np
import scipy.fftpack
import skimage
import os
import skimage.io
# import h5py
import pylab

from random import randint

imageSize = [256,256]

def getSampleTopLeftCorner (iMin,iMax,jMin,jMax):
    """ Function that generates randomly a position between i,j intervals [iMin,iMax], [jMin,jMax]
    Args:
        iMin (int): the i minimum coordinate (i is the column-position of an array)
        iMax (int): the i maximum coordinate (i is the column-position of an array)
        jMin (int): the j minimum coordinate (j is the row-position of an array)
        jMax (int): the j maximum coordinate (j is the row-position of an array)
    Returns:
        [i,j] (tuple(int,int)): random integers such iMin<=i<iMax,jMin<=j<jMax,
    """
    i = randint(iMin, iMax)
    j = randint(jMin, jMax)
    return [i, j]

def getSampleImage (image, sampleSize, topLeftCorner):
    """ Function that extracts a sample of an image with a given size and a given position
    Args:
        image (numpy.array) : input image to be sampled
        sampleSize (tuple(int,int)): size of the sample
        topLeftCorner (tuple(int,int)): positon of the top left corner of the sample within the image
    Returns:
        sample (numpy.array): image sample
    """
    return image[topLeftCorner[0]:topLeftCorner[0] + sampleSize[0], topLeftCorner[1]:topLeftCorner[1] + sampleSize[1]]

def getRandomSampleImage (image, sampleSize):
    Xmax = imageSize[0] - sampleSize[0]
    Ymax = imageSize[1] - sampleSize[1]
    return getSampleImage(image, sampleSize, getSampleTopLeftCorner(0, Xmax, 0, Ymax))

def getSamplePS (sample):
    """ Function that calculates the power spectrum of a image sample
    Args:
        sample (numpy.array): image sample
    Returns:
        samplePS (numpy.array): power spectrum of the sample. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """
    F1 = scipy.fftpack.fft2(sample)
    F2 = np.abs(scipy.fftpack.fftshift(F1)**2)
    return np.matrix(F2)

def getAveragePS (inputDirectory, sampleSize):
    """ Function that estimates the average power spectrum of a image database
    Args:
        inputDirectory (str) : Absolute pathway to the image database
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
    Returns:
        averagePS (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """
    averagePS = np.zeros(shape=sampleSize)
    listOfFiles = os.listdir(inputDirectory)
    for i in range(len(listOfFiles)):
        inputFileName = inputDirectory + listOfFiles[i]
        print inputFileName
        img = skimage.io.imread(inputFileName)
        smp = getRandomSampleImage(img, sampleSize)
        averagePS += getSamplePS(smp)
    return averagePS / len(listOfFiles)


def getRadialFreq(PSSize):
    """ Function that returns the Discrete Fourier Transform radial frequencies
    Args:
        psSize (tuple(int,int)): the size of the window to calculate the frequencies
    Returns:
        radialFreq (numpy.array): radial frequencies in crescent order
    """
    fx = np.fft.fftshift(np.fft.fftfreq(PSSize[0], 1./PSSize[0]));
    fy = np.fft.fftshift(np.fft.fftfreq(PSSize[1], 1./PSSize[1]));
    [X,Y] = np.meshgrid(fx,fy);
    R = np.sqrt(X**2+Y**2);
    radialFreq = np.unique(R);
    radialFreq.sort()
    return radialFreq[radialFreq!=0]

def getRadialPS(averagePS):
    """ Function that estimates the average power radial spectrum of a image database
    Args:
        averagePS (numpy.array) : average power spectrum of the database samples.
    Returns:
        averagePSRadial (numpy.array): average radial power spectrum of the database samples.
    """
    size = averagePS.shape
    radialFreq = getRadialFreq(size)
    amount = dict( (f, 0) for f in radialFreq )
    value = dict( (f, 0) for f in radialFreq )
    fx = np.fft.fftshift(np.fft.fftfreq(size[0], 1./size[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(size[1], 1./size[1]))
    for x in range(size[0]):
        for y in range(size[1]):
            f = np.sqrt(fx[x]**2 + fy[y]**2)
            if (f != 0):
                amount[f] += 1
                value[f] += averagePS[x, y]
    return [value[f] / amount[f] for f in radialFreq]


def getAveragePSLocal(inputDirectory, sampleSize, gridSize):
    """ Function that estimates the local average power spectrum of a image database
    Args:
        inputDirectory (str) : Absolute pathway to the image database
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
        gridSize (tuple(int,int)): size of the grid that define the borders of each local region
    Returns:
        averagePSLocal (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see numpy.fft.fftshift)
    """
    return averagePSLocal

def getAveragePSLocal(inputDirectory, sampleSize, gridSize):
    """ Function that estimates the local average power spectrum of a image database
    Args:
        inputDirectory (str) : Absolute pathway to the image database
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
        gridSize (tuple(int,int)): size of the grid that define the borders of each local region
    Returns:
        averagePSLocal (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see numpy.fft.fftshift)
    """
    ###write your function here


def makeAveragePSFigure (averagePS, figureFileName):
    """ Function that makes and save the figure with the power spectrum
    Args:
        averagePSLocal (numpy.array): the average power spectrum in an array of shape [sampleShape[0],sampleShape[1]
        figureFileName (str): absolute path where the figure will be saved
    """
    pylab.imshow(np.log(averagePS), cmap = "gray")
    # pylab.contour(np.log(averagePS))
    pylab.axis("off")
    pylab.savefig(figureFileName)

def makeAveragePSRadialFigure (radialFreq,averagePSRadial,figureFileName):
    """ Function that makes and save the figure with the power spectrum
    Args:
        averagePS (numpy.array) : the average power spectrum
        averagePSRadial (numpy.array): the average radial power spectrum
        figureFileName (str): absolute path where the figure will be saved
    """
    pylab.figure()
    pylab.loglog(radialFreq, averagePSRadial, '.')
    pylab.xlabel("Frequecy")
    pylab.ylabel("Radial Power Spectrum")
    pylab.savefig(figureFileName)


def makeAveragePSLocalFigure(averagePSLocal,figureFileName,gridSize):
    """ Function that makes and save the figure with the local power spectrum
    Args:
        averagePSLocal (numpy.array): the average power spectrum in an array of shape [gridSize[0],gridSize[1],sampleShape[0],sampleShape[1]
        figureFileName (str): absolute path where the figure will be saved
        gridSize (tuple): size of the grid
    """
    pylab.figure()
    for i in range(gridSize[0]):
        for j in range(gridSize[1]):
            pylab.subplot(gridSize[0],gridSize[1],i*gridSize[1]+j+1)
            pylab.imshow(np.log(averagePSLocal[i,j]),cmap = "gray")
            pylab.contour(np.log(averagePSLocal[i,j]))
            pylab.axis("off")
    pylab.savefig(figureFileName)

def saveH5(fileName,dataName,numpyArray):
    """ Function that saves numpy arrays in a binary file h5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataName (str): the dataset name
        numpyArray (numpy.array): the data to be saved
    """

    # f = h5py.File(fileName, "w")
    # f.create_dataset(dataName,data =numpyArray);
    # f.close()


def readH5(fileName, dataName):
    """ Function that reads numpy arrays in a binary file hdf5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataName (str): the dataset name
    Return:
        numpyArray (numpy.array): the read data
    """
    # f = h5py.File(fileName, "r")
    # numpyArray = f[dataName][:]
    # return numpyArray
