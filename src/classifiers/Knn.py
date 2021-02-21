
from src.metrics.Euclidean import Euclidean
from sklearn.neighbors import KNeighborsClassifier
from src.classifiers.Classifier import Classifier

class Knn(Classifier):

  def __init__(self, train, test, labelCol, n_neighbors):
    super(Knn, self)
    xOff, yOff, xOn, yOn = super()._preProcess(train, test, labelCol, avgTrain=False)
    self.__classifier(xOff, yOff, xOn, yOn, n_neighbors)
  
  def __classifier(self, xOff, yOff, xOn, yOn, n_neighbors):
    clf = self.__fit(xOff, yOff, n_neighbors)
    euclidean = Euclidean()
    
    yPred = clf.predict(xOn)
    euclidean.compute(yOn, yPred, pos=False)
    self._distances = euclidean.getDistances()
    self._predictions = yPred
    self._realLabels = yOn
     
  def __fit(self, xOff, yOff, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(xOff, yOff)
    return clf
  
  

    

