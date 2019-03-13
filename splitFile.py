import pandas as pd
import os
from pathlib import Path

file = 'capture20110810Resampled.csv'

dataFrame = pd.read_csv(file)

totalRows = len(dataFrame.index)

dataFrame1 = dataFrame[0:1050000]
dataFrame2 = dataFrame[1050000:]

dataFrameOutDir = Path("Output")
dataFrameOutDir.mkdir(parents=True, exist_ok=True)

dataframeOutName1 = file[:-4] + "Part1.csv"
dataframeOutName2 = file[:-4] + "Part2.csv"

dataFrame1.to_csv(dataFrameOutDir / dataframeOutName1, encoding='utf-8', index=False)
print(dataframeOutName1 + " Created \n")

dataFrame2.to_csv(dataFrameOutDir / dataframeOutName2, encoding='utf-8', index=False)
print(dataframeOutName2 + " Created \n")