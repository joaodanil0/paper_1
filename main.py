
from src.loads.TrainBase import TrainBase
from src.loads.TestBase import TestBase
from src.commom.Util import Util

trainPath = "bases/train/"
testPath = "bases/test/"

# util = Util()
train = TrainBase(trainPath)
test = TestBase(testPath)

test.setLabels(train.getLabels("LABEL"), "LABEL")

# firstBase = list(train.getBases().values())[0]
# parse = util.createMapPositionByLabel(firstBase, "LABEL", "X", "Y")



