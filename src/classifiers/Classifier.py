

import numpy as np
from src.commom import Info

from src.metrics.Euclidean import Euclidean
from sklearn.metrics import accuracy_score

class Classifier(object):

  __distances = None
  __predictions = None
  __realLabels = None

  def __init__(self, train, test, labelCol, **kwargs):
    xOff, yOff, xOn, yOn = self.__preProcess(train, test, labelCol)
    self.__classifier(xOff, yOff, xOn, yOn, **kwargs)

  def __preProcess(self, train, test, labelCol):
    xOff = train[Info.cols]
    yOff = train[labelCol]
    xOn = test[Info.cols]
    yOn = test[labelCol]
    return xOff, yOff, xOn, yOn  

  def __classifier(self, xOff, yOff, xOn, yOn, **kwargs):
    clf = self._fit(xOff, yOff, **kwargs)
    
    yPred = clf.predict(xOn)
    euclidean = Euclidean(yOn, yPred)
    self.__distances = euclidean.getDistances()
    self.__predictions = yPred
    self.__realLabels = yOn
  
  def __getRooms(self, labels):
    return [point.split("_")[0] for point in labels]

  def getDistances(self):
    return self.__distances
  
  def getPredictions(self):
    return self.__predictions
  
  def getTrueLabels(self):
    return np.array(self.__realLabels)
  
  def getPointAccuracy(self):
    return accuracy_score(self.getTrueLabels(), self.getPredictions())*100
  
  def getRoomAccuracy(self):
    predRooms = self.__getRooms(self.getPredictions())
    realRooms = self.__getRooms(self.getTrueLabels())
    return accuracy_score(realRooms, predRooms)*100

