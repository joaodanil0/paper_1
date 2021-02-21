
class Util:

  def __init__(self):
    pass
  
  def createMapPositionByLabel(self, base, labelCol, xCol, yCol):
    parse = {}
    labels = base[labelCol].unique()
    for label in labels:
      subBase = base[base.LABEL == label]
      sample = subBase.head(1)
      parse[label] = (float(sample[xCol].values[0]), float(sample[yCol].values[0]))
    print(parse)
    return parse