
import time
import pandas as pd
from src.loads.TrainBase import TrainBase
from src.loads.TestBase import TestBase

from src.classifiers.Knn import Knn
from src.classifiers.Wknn import Wknn
from src.classifiers.Pa import Pa

trainPath = "bases/train/synth" + "/"
testPath  = "bases/test/original_test" + "/"

train = TrainBase(trainPath)
test = TestBase(testPath)

test.setLabels(train.getLabels("LABEL"), "LABEL")

nameFiles = []
avgErrors = []
stdErrors = []
pointAccs = []
roomAccs = []
dropeds = []

i = 0
totalTime = 0

for trainNameFile, trainFile in train.getBases().items():
  i+= 1
  for testNameFile, testFile in test.getBases().items():
    
    start_time = time.time()

    clf = Pa(trainFile, testFile, "LABEL", 0.01)
    dists = clf.getDistances()
    pointAcc = clf.getPointAccuracy()
    roomAcc = clf.getRoomAccuracy()
    dropedLines = clf.getDropedLines()
    
    # pd.Series(dists).to_csv("results/Pa_best_synth_errors_wt_0.01.csv", index=False)
    avgError = dists.mean()
    stdError = dists.std()

    nameFiles.append(trainNameFile)
    avgErrors.append(avgError)
    stdErrors.append(stdError)
    pointAccs.append(pointAcc)
    roomAccs.append(roomAcc)
    dropeds.append(len(dropedLines))

    end_time = time.time()
    
    print("[%d] "
          "AVG error: %.2f, STD error: %.2f, "
          "Point acc: %.2f, Room acc: %.2f, "
          "Droped lines: %d, "
          "Execution time: %.2f s" % (i, avgError, stdError,
                                      pointAcc, roomAcc, len(dropedLines), 
                                      end_time - start_time)) 
    totalTime += (end_time - start_time)

aux = pd.DataFrame({
  "Files": nameFiles, 
  "AVG error (m)": avgErrors, 
  "STD error (m)": stdErrors,
  "Acc point (%)": pointAccs, 
  "Acc room (%)":roomAccs,
  "Droped lines": dropeds})
aux.to_csv("results/Results_Pa.csv", index=False, float_format="%.2f")
print("Acabou em: %.2f" % totalTime)




