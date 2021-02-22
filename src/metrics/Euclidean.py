
import numpy as np
from scipy.spatial import distance

from src.commom import Info

class Euclidean:

  __distances = None
  
  def labelToPosition(self, labels):
    return [Info.positionByLabelMap[label] for label in labels]
  
  def __computeDistance(self, realPos, predPos):    
    lenList = len(realPos)
    distances = []
    for i in range(lenList):
      distances.append(distance.euclidean(realPos[i], predPos[i]))    
    self.__distances = np.array(distances)
  
  def compute(self, realPos, predPos, pos=True):
    if pos:
      self.__computeDistance(realPos, predPos)
    else:
      realPos = self.labelToPosition(realPos)
      predPos = self.labelToPosition(predPos)
      self.__computeDistance(realPos, predPos)

  def getDistances(self):
    return self.__distances
  
  def getDistanceBetweenVectors(self, vectorA, vectorB):
    return distance.cdist(vectorA, vectorB, 'euclidean')



