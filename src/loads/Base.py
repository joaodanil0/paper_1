
import os

import pandas as pd

from glob import glob

class Base(object):

  __bases = None 

  def __init__(self, rootPath):
    pathFiles = self.__loadCsvFileNames(rootPath)
    self.__bases = self.__loadAllCsvFiles(pathFiles)

  def __loadCsvFileNames(self, path):    
    return sorted(glob(path + '**/*.csv', recursive=True))
  
  def _loadCsvFile(self, pathFile):
    return pd.read_csv(pathFile)
  
  def __extractNameFile(self, pathFile):
    return os.path.splitext(pathFile.split("/")[-1])
  
  def __loadAllCsvFiles(self, pathFiles):
    bases = {}
    for pathFile in pathFiles:
      nameFile = self.__extractNameFile(pathFile)[0]
      file = self._loadCsvFile(pathFile)
      bases[nameFile] = file    
    return bases
  
  def getBases(self):
    return self.__bases
  
  def getBasesName(self):
    return list(self.__bases.keys())
  
  def getLabels(self, labelCol):
    return self.__bases[self.getBasesName()[0]][labelCol].unique()

  def setLabels(self, labels, labelCol):
    bases = {}
    baseNames = []
    for nameFile, file in self.__bases.items():
      bases[nameFile] = file[file[labelCol].isin(labels)]
    self.__bases = bases

