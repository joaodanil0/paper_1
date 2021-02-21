
from src.classifiers.Classifier import Classifier
from src.commom import Info

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from src.metrics.Euclidean import Euclidean

class Wknn(Classifier):

  def __init__(self, train, test, labelCol, n_neighbors):
    super(Wknn, self)
    xOff, yOff, xOn, yOn = super()._preProcess(train, test, labelCol, avgTrain=True)
    self.__classifier(xOff, yOff, xOn, yOn, n_neighbors)
  
  def __fit(self, xOff, yOff, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(xOff, yOff)
    return clf

  def __classifier(self, xOff, yOff, xOn, yOn, n_neighbors):

    clf = self.__fit(xOff, yOff, n_neighbors)
    euclidean = Euclidean()
    
    dists, idxs = clf.kneighbors(xOn)
    predPos = self.__computePosition(dists, idxs, yOff)    
    realPos = euclidean.labelToPosition(yOn)

    euclidean.compute(realPos, predPos)
    self._distances = euclidean.getDistances()
    self._predictions = self.__getClosestLabel(predPos)
    self._realLabels = yOn
  
  def __computePosition(self, dists, idxs, yOff): 
    dists += np.finfo(float).eps
    sum_threshold = np.reciprocal(dists).sum(axis=1)
    pred_positions = np.empty(idxs.shape, dtype=object)

    for row in range(idxs.shape[0]):
      for col in range(idxs.shape[1]):
        pred_positions[row,col] = np.array(Info.positionByLabelMap[yOff[idxs[row,col]]])
    
    return (np.reciprocal(sum_threshold)*(np.reciprocal(dists)*pred_positions).sum(axis=1)).tolist()

  def __getClosestLabel(self, predPos):
    labels = np.array(list(Info.positionByLabelMap.keys()))
    positions = np.array(list(Info.positionByLabelMap.values()))
    NN = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(positions) 
    distances, indices = NN.kneighbors(predPos)
    return [labels[idx] for idx in indices.flatten()]