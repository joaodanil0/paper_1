
from src.classifiers.Classifier import Classifier
from src.metrics.Euclidean import Euclidean
from src.commom import Info
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import numpy as np

class Pa(Classifier):

  def __init__(self, train, test, labelCol, wt):
    super(Pa, self)
    xOff, yOff, xOn, yOn = super()._preProcess(train, test, labelCol, avgTrain=True)
    self.__classifier(xOff, yOff, xOn, yOn, wt)

  def __classifier(self, xOff, yOff, xOn, yOn, wt):
    
    euclidean = Euclidean()    

    predPos = self.__computePosition(xOff, yOff, xOn, wt, euclidean)  
    realPos = euclidean.labelToPosition(yOn)

    euclidean.compute(realPos, predPos)
    self._distances = euclidean.getDistances()
    self._predictions = self.__getClosestLabel(predPos)
    self._realLabels = yOn
  
  def __computePosition(self, xOff, yOff, xOn, wt, euclidean):
    xOffStandardized = self.__standardizer(xOff)
    xOnStandardized = self.__standardizer(xOn)

    d_i = euclidean.getDistanceBetweenVectors(xOnStandardized, xOffStandardized)
    sum_d_i = self.__piecewiseInverse(d_i).sum(axis=1) 
    w_i = self.__divideMatrixColsByVector(self.__piecewiseInverse(d_i), sum_d_i)
    w_th = self.__putZerosUnderThreshold(w_i, wt)

    new_di = self.__multipliesMatrixColsByVector(w_th, sum_d_i)
    new_sum_d_i = new_di.sum(axis=1)

    """
    Adjust:
      
      The "new_di" is a matrix of shape (148, 12097)
            148 -> Number of train labels
            12097 -> Number of test samples
      The objective is that "positions" has the same shape to multiplie with "new_di"
      but each value of "positions" is a tuple with position X and Y

      The solution applied here was:
        - Flatten the "new_di", putting each column under of each other (resulting in shape: (1790356,))
        - Generate a position vector (train labels 148 positions) replicated 12097 (number of test samples)
           (resulting in shape: (1790356, 2) the 2 cols is due X and Y positions
        - Multiplie row piece wise 
        - Reshape to "new_di" shape, but with X and Y positions
    """
    xOnNumRows = xOn.shape[0]
    xOffNumRows = xOff.shape[0]
    positionsOff = euclidean.labelToPosition(yOff)
    positions = self.__concatenateList(positionsOff, xOnNumRows)
    
    positionsNumCols = positions.shape[1]
      
    flatten_inv_new_di = self.__transformTo1dArray(new_di)
    a = self.__multipliesMatrixColsByVector(positions, flatten_inv_new_di.T)
    a = self.__transformToMatrix(a, xOnNumRows,xOffNumRows,positionsNumCols)

    result = a.sum(axis=1)  
    return np.reciprocal(new_sum_d_i)[:, np.newaxis] * result

  def __standardizer(self, vector):
    return StandardScaler().fit_transform(vector.values.T).T
  
  def __divideMatrixColsByVector(self, matrix, vector):
    return  np.divide(matrix, vector[:, np.newaxis])
  
  def __multipliesMatrixColsByVector(self, matrix, vector):
    return np.multiply(matrix, vector[:, np.newaxis])
  
  def __piecewiseInverse(self, matrix):
    return np.reciprocal(matrix)
  
  def __putZerosUnderThreshold(self, matrix, threshold):    
    newMatrix = np.where(matrix > threshold, matrix, 0)
    newMatrix += np.finfo(float).eps # add infinitesimal value (close to zero)
    return newMatrix
  
  def __transformTo1dArray(self, matrix):
    return np.array(matrix).flatten() 
  
  def __transformToMatrix(self, vector, dim1, dim2, dim3):
    return np.asarray(vector).reshape(dim1,dim2,dim3)
  
  def __concatenateList(self, list_, amountOfConcat):
    return np.array(list_*amountOfConcat)

  def __getClosestLabel(self, predPos):
    labels = np.array(list(Info.positionByLabelMap.keys()))
    positions = np.array(list(Info.positionByLabelMap.values()))
    NN = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(positions) 
    distances, indices = NN.kneighbors(predPos)
    return [labels[idx] for idx in indices.flatten()]