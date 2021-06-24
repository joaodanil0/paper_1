import json
import time
import pandas as pd
import numpy as np

from src.classifiers.Pa import Pa
from src.loads.TestBase import TestBase

def loadJson(path):
  with open(path) as json_file:
    return json.load(json_file)

jsonPath = 'results/10_best.json'
jsonData = loadJson(jsonPath)

trainPath = "bases/train/synth" + "/"
testPath  = "bases/test/original_test" + "/"

test = TestBase(testPath)
testFile = list(test.getBases().values())[0]

for tick_num, tick_values in jsonData.items():
  nameFiles = []
  avgErrors = []
  stdErrors = []
  pointAccs = []
  roomAccs = []
  dropeds = []

  i = 0
  totalTime = 0
  for bases in tick_values:
    start_time = time.time()
    stock_files = [trainPath + file + ".csv" for file in bases]
    
    trainFile = pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)
    clf = Pa(trainFile, testFile, "LABEL", 0.01)

    dists = clf.getDistances()
    pointAcc = clf.getPointAccuracy()
    roomAcc = clf.getRoomAccuracy()
    dropedLines = clf.getDropedLines()

    avgError = dists.mean()
    stdError = dists.std()

    nameFiles.append(tick_num)
    avgErrors.append(avgError)
    stdErrors.append(stdError)
    pointAccs.append(pointAcc)
    roomAccs.append(roomAcc)
    dropeds.append(len(dropedLines))

    end_time = time.time()
    
    # print("[%d] "
    #       "AVG error: %.2f, STD error: %.2f, "
    #       "Point acc: %.2f, Room acc: %.2f, "
    #       "Droped lines: %d, "
    #       "Execution time: %.2f s" % (int(tick_num), avgError, stdError,
    #                                   pointAcc, roomAcc, len(dropedLines), 
    #                                   end_time - start_time)) 
    totalTime += (end_time - start_time)
  print("[%s] AVG error: %.2f, STD error: %.2f" % (tick_num, np.array(avgErrors).mean(), np.array(avgErrors).std()))