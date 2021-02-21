
import pandas as pd
from src.loads.Base import Base

class TestBase(Base):
  
  def __init__(self, testath):
    super(TestBase, self).__init__(testath)
  
  def _loadCsvFile(self, pathFile):
    base = pd.read_csv(pathFile)
    base["ROOM_ID"] = base["ROOM_ID"].astype(int)
    base = base.rename(columns={"ROOM_ID": "ROOM"})
    return base
