import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

"""
    Script to slightly adjust the label encoded features to properly encode them considering the entire data
    Allows for a fix from locally encoded values to globally encoding the values
"""

localFiles = [file for file in os.listdir('.') if file.endswith(".csv")]

#Iterate through the two merged files (testing file and training file)
for file in localFiles:
    print("Generating features for file: "+file)
    dataFrame = pd.read_csv(file)

    #Drop all previous incorrectly computed columns
    #Note: State is dropped and not recomputed --> Is not used for training
    colsToDrop = ["State","TotBytesDisc","SrcBytesDisc","SportDisc","SportDisc","DportDisc","Src_TotBytesDisc_mode","Dst_TotBytesDisc_mode"]

    dataFrame.drop(colsToDrop, axis=1, inplace=True)

    quantile_list = [0, .25, .5, .75, 1.] # Change the quantile_list for more or less accuracy

    dataFrame['TotBytesDisc'] = ""
    dataFrame['SrcBytesDisc'] = ""
    dataFrame['TotBytesDisc'] = pd.qcut(dataFrame['TotBytes'], quantile_list)
    dataFrame['SrcBytesDisc'] = pd.qcut(dataFrame['SrcBytes'], quantile_list)

    #Label encode discretized byte vals
    le = preprocessing.LabelEncoder()
    le.fit(dataFrame.TotBytesDisc.unique())
    dataFrame.TotBytesDisc = le.transform(dataFrame.TotBytesDisc)

    le = preprocessing.LabelEncoder()
    le.fit(dataFrame.SrcBytesDisc.unique())
    dataFrame.SrcBytesDisc = le.transform(dataFrame.SrcBytesDisc)

    dataFrame['SportDisc'] = ""
    dataFrame['DportDisc'] = ""

    dataFrame['SportDisc'] = pd.cut(dataFrame['Sport'],[0,1023,49151,65535],labels=[0,1,2])
    dataFrame['DportDisc'] = pd.cut(dataFrame['Dport'],[0,1023,49151,65535],labels=[0,1,2])


    #Outputting file
    dataFrameOutDir = Path("NewFeatureGenerated")
    dataFrameOutDir.mkdir(parents=True, exist_ok=True)
    
    dataframeOutName = file[:-4] + "NewFeatureGenerated.csv"

    dataFrame.to_csv(dataFrameOutDir / dataframeOutName, encoding='utf-8', index=False)
    print(dataframeOutName + " Created")