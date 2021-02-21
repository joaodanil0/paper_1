
import numpy as np
from scipy.spatial import distance

from src.commom import Info

class Euclidean:

  __distances = None

  def __init__(self, yReal, yPred):
    self.__computeDistance(yReal, yPred)
  
  def __labelToPosition(self, labels):
    return [Info.positionByLabelMap[label] for label in labels]

  def __computeDistance(self, yReal, yPred):
    
    lenList = len(yReal)
    realPos = self.__labelToPosition(yReal)
    predPos = self.__labelToPosition(yPred)

    distances = []
    for i in range(lenList):
      distances.append(distance.euclidean(realPos[i], predPos[i]))    
    self.__distances = np.array(distances)
  
  def getDistances(self):
    return self.__distances

