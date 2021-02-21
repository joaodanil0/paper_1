
import pandas as pd
from src.loads.Base import Base

class TrainBase(Base):
  
  _baseNames = []
  def __init__(self, trainPath):
    super(TrainBase, self).__init__(trainPath)

  def _loadCsvFile(self, pathFile):
    base = pd.read_csv(pathFile)
    base["ROOM"] = base.LABEL.str.split("_", 1).str[0]
    base["ROOM"] = base["ROOM"].astype(int)
    return base
  
  
