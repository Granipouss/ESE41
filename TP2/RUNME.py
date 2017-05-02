import sys
TD1AbsolutePath = "../TP1/"
sys.path.append(TD1AbsolutePath)

import PSpy
import WhiteICA
import numpy as np


inputDirectory = "../img/"
resultsDirectory = "./res/"
resultsTD1Directory = "../TP/res/"

averagePSResultsFileName = resultsTD1Directory + "averagePS.hdf5"

whiteningFiltersFigureFileName = resultsDirectory + "whiteningFilters.png"
whiteningFiltersResultsFileName = resultsDirectory + "whiteningFilters.hdf5"

averagePS1FigureFileName = resultsDirectory +"averagePS1.png"
averagePS4FigureFileName = resultsDirectory +"averagePS4.png"

averagePS1ResultsFileName = resultsDirectory +"averagePS1.hdf5"
averagePS4ResultsFileName = resultsDirectory +"averagePS4.hdf5"

averagePS1RadialResultsFileName = resultsDirectory + "averagePS1Radial.hdf5"
averagePS4RadialResultsFileName = resultsDirectory + "averagePS4Radial.hdf5"

averagePS1RadialFigureFileName = resultsDirectory + "averagePS1Radial.png"
averagePS4RadialFigureFileName = resultsDirectory + "averagePS4Radial.png"

ICResultsFileName = resultsDirectory +  "IC.hdf5"
ICFigureFileName = resultsDirectory + "IC.png"

sampleSize = [64,64]
ICASampleSize = [12,12]
ICANSamples = 50000;

averagePS = PSpy.readH5(averagePSResultsFileName,'averagePS')

maxPS = np.max(averagePS);
noiseVarianceList = [maxPS*10**(-9),maxPS*10**(-8),maxPS*10**(-7),maxPS*10**(-6)]


whiteningFilters = [];
for noiseVariance in noiseVarianceList:
    whiteningFilters.append(Whitepy.getPowerSpectrumWhiteningFilter(averagePS,noiseVariance))

PSpy.saveH5(whiteningFiltersResultsFileName,'whiteningFilters',np.array(whiteningFilters))
Whitepy.makeWhiteningFiltersFigure(whiteningFilters,whiteningFiltersFigureFileName)

whiteningFilter = whiteningFilters[1];
averagePS1 = Whitepy.getAveragePSWhitenImages(inputDirectory, sampleSize,whiteningFilter)

PSpy.saveH5(averagePS1ResultsFileName,'averagePS',averagePS1)
PSpy.makeAveragePSFigure(averagePS1, averagePS1FigureFileName)

averagePS1Radial = PSpy.getRadialPS(averagePS1)
radialFreq = PSpy.getRadialFreq(averagePS1.shape)
PSpy.saveH5(averagePS1RadialResultsFileName,'averagePSRadial',averagePS1Radial)
PSpy.makeAveragePSRadialFigure(radialFreq,averagePS1Radial, averagePS1RadialFigureFileName)


X = Whitepy.getICAInputData(inputDirectory, ICASampleSize, ICANSamples)
X = Whitepy.preprocess(X);
C = Whitepy.getIC(X)

PSpy.saveH5(ICResultsFileName,'IC',C)
Whitepy.makeIdependentComponentsFigure(C,ICASampleSize, ICFigureFileName)
