

import numpy as np
from src.commom import Info

from sklearn.metrics import accuracy_score

class Classifier(object):

  _distances = None
  _predictions = None
  _realLabels = None
  _dropedLines = []
 
  def _preProcess(self, train, test, labelCol, avgTrain=False):

    if avgTrain:
      train = train.groupby([labelCol]).mean().reset_index()
      
    xOff = train[Info.cols]
    yOff = train[labelCol]
    xOn = test[Info.cols]
    yOn = test[labelCol]
    return xOff, yOff, xOn, yOn  
  
  def __getRooms(self, labels):
    return [point.split("_")[0] for point in labels]

  def getDistances(self):
    return self._distances
  
  def getPredictions(self):
    return self._predictions
  
  def getTrueLabels(self):
    return np.array(self._realLabels)
  
  def getPointAccuracy(self):
    return accuracy_score(self.getTrueLabels(), self.getPredictions())*100
  
  def getDropedLines(self):
    return self._dropedLines
  
  def getRoomAccuracy(self):
    predRooms = self.__getRooms(self.getPredictions())
    realRooms = self.__getRooms(self.getTrueLabels())
    return accuracy_score(realRooms, predRooms)*100

