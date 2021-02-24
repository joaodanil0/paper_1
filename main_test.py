
from src.loads.TrainBase import TrainBase
from src.classifiers.Pa import Pa

from src.commom import Info
import numpy as np
import pandas as pd
import json


trainPath = "bases/train/synth" + "/"

train = TrainBase(trainPath)

ticks = np.linspace(1,len(Info.labels), 20).astype(int)
random_samples = np.arange(30)

dic_tick = {}
for k, tick in enumerate(ticks):
  trains = {}  
  for trainNameFile, trainFile in train.getBases().items():
    steps = []
    for random_sample in random_samples:      
      testFile = pd.read_csv("bases/test/sample_tests/" + str(random_sample) + "/" + str(tick) + ".csv")
      clf = Pa(trainFile, testFile, "LABEL", 0.01)
      dists = clf.getDistances()
      steps.append(dists.mean())
    trains[str(trainNameFile)] = steps
  dic_tick[str(tick)] = trains

with open('results/Pa.json', 'w') as fp:
  json.dump(dic_tick, fp)
