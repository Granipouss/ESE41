import sys
TD1AbsolutePath = "/put/your/path/toPSpy.py/here"
sys.path.append(TD1AbsolutePath)
import os
import numpy as np
import PSpy
import pylab
import scipy.ndimage.filters as filters
import skimage.io
from sklearn.decomposition import FastICA


imageSize = [256, 256]

def truncateNonNeg (X):
    """Function that truncates arrays od real numbers into arrays of non negatives.
    Args:
        X(numpy.array): input array
    Returns:
        Y(numpy.array): array with positive or zero numbers
    """
    return X.clip(0)

def getPowerSpectrumWhiteningFilter (averagePS, noiseVariance):
    """Function that estimates the whitening and denoising power spectrum filter
    Args:
        averagePS(numpy.array): average power spectrum of the observation
        noiseVariance(double): variance of the gaussian white noite.
    Returns:
        w(numpy.array): whitening denoising filter
    """
    sM = noiseVariance * (averagePS.shape[0] * averagePS.shape[1])
    ps = np.fft.ifftshift(averagePS)
    denoising = ((ps - sM) / ps).clip(0)
    whitening = (1 / np.sqrt(ps))
    F = whitening * denoising
    F = np.fft.ifft2(F)
    F = np.real(F)
    F = np.fft.fftshift(F)
    return F

def getAveragePSWhitenImages (inputDirectory, sampleSize, whiteningFilter):
    """ Function that estimates the average power spectrum of a image database
    Args:
        inputDirectory (str) : Absolute pathway to the image database
        sampleSize (tuple(int, int)): size of the samples that are extrated from the images
        whiteningFilter (numpy.array) : whitening filter
    Returns:
        averagePS (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """
    averagePS = np.zeros(shape=sampleSize)
    listOfFiles = os.listdir(inputDirectory)
    for i in range(len(listOfFiles)):
        inputFileName = inputDirectory + listOfFiles[i]
        # print inputFileName
        img = skimage.io.imread(inputFileName)
        smp = PSpy.getRandomSampleImage(img, sampleSize)
        smp = filters.convolve(smp, whiteningFilter, mode='wrap')
        # smp = whiteningFilter * smp
        averagePS += PSpy.getSamplePS(smp)
    return averagePS / len(listOfFiles)

def getICAInputData (inputDirectory, sampleSize, nSamples):
    """ Function that samples the input directory for later to be used by FastICA
    Args:
        inputDirectory(str):: Absolute pathway to the image database
        sampleSize (tuple(int, int)): size of the samples that are extrated from the images
        nSamples(int): number of samples that should be taken from the database
    Returns:
        X(numpy.array)nSamples, sampleSize
    """
    listOfFiles = os.listdir(inputDirectory)
    nImages = len(listOfFiles)
    X = np.zeros([nSamples,] + sampleSize)
    for i in range(nSamples):
        fileName = np.random.choice(listOfFiles, 1)[0]
        fullFileName = inputDirectory + fileName
        print fullFileName
        im = skimage.io.imread(fullFileName)
        samplePos = PSpy.getSampleTopLeftCorner(0, imageSize[0] - sampleSize[0], 0, imageSize[1] - sampleSize[1]);
        sample = PSpy.getSampleImage(im, sampleSize, samplePos).astype("double") / 256.;
        X[i] = sample
    return X;

def preprocess (X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
        qsqX(numpy array): input to be preprocessed
    Returns:
        qsqX(numpy.array)
    """
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])

def getIC (X):
    """Function that estimates the principal components of the data
    Args:
        X(numpy.array):preprocessed data
    Returns:
        S(numpy.array) the matrix of the independent sources of the data
    """
    ica = FastICA(n_components=X.shape[0], algorithm='parallel', whiten=True)
    ica.fit_transform(X)
    return ica.components_

def makeWhiteningFiltersFigure (whiteningFilters, figureFileName):
    pylab.figure()
    for i, whiteningFilter in enumerate(whiteningFilters):
        pylab.subplot(1, len(whiteningFilters), i + 1)
        vmax = np.max(np.abs(whiteningFilter))
        vmin = -vmax
        pylab.imshow(whiteningFilter, cmap='gray', vmax = vmax, vmin = vmin)
        pylab.axis("off")
    pylab.savefig(figureFileName)

def makeIdependentComponentsFigure (C, sampleSize, figureFileName):
    C = C.reshape([-1,] + sampleSize)
    pylab.figure()
    # for i in range(np.min([C.shape[0], sampleSize[0] * sampleSize[1]])):
    for i in range(C.shape[0]):
        pylab.subplot(sampleSize[0], sampleSize[1], i + 1)
        pylab.imshow(C[i], cmap='gray')
        pylab.axis("off")
    pylab.savefig(figureFileName)
