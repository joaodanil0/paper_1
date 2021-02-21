
from sklearn.neighbors import KNeighborsClassifier
from src.classifiers.Classifier import Classifier

class Knn(Classifier):

  def __init__(self, train, test, labelCol, n_neighbors):
    super(Knn, self).__init__(train, test, labelCol, n_neighbors=n_neighbors)
  
  def _fit(self, xOff, yOff, **kwargs):
    n_neighbors = kwargs.get('n_neighbors')
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(xOff, yOff)
    return clf
     

    

