
import time
import pandas as pd
from src.loads.TrainBase import TrainBase
from src.loads.TestBase import TestBase

from src.classifiers.Knn import Knn
from src.classifiers.Wknn import Wknn
from src.classifiers.Pa import Pa

trainPath = "bases/train/"
testPath = "bases/test/"

train = TrainBase(trainPath)
test = TestBase(testPath)

test.setLabels(train.getLabels("LABEL"), "LABEL")


nameFiles = []
avgErrors = []
pointAccs = []
roomAccs = []

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

    avgError = dists.mean()
  
    nameFiles.append(trainNameFile)
    avgErrors.append(dists.mean())
    pointAccs.append(pointAcc)
    roomAccs.append(roomAcc)

    end_time = time.time()
    
    print("[%d] File: %s, AVG error: %.2f, Point acc: %.2f, Room acc: %.2f, Execution time: %.2f s" % (  i, trainNameFile, 
                                                                          avgError,
                                                                          pointAcc, 
                                                                          roomAcc,                                                                          
                                                                          end_time - start_time)
                                                                        ) 
    totalTime += (end_time - start_time)

aux = pd.DataFrame({"Files": nameFiles, "AVG error (m)": avgErrors, "Acc point (%)": pointAccs, "Acc room (%)":roomAccs})
aux.to_csv("Results_Pa.csv", index=False, float_format="%.2f")
print("Acabou em: %.2f" % totalTime)




