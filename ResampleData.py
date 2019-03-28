import pandas as pd
import os
import math
from pathlib import Path

"""
This script will take all raw data Files and resample them as follows:
If first botnet at >20k then take 20k background rows behind botnet
Then take every background row between 2 botnet rows is the botnet rows <20k apart
If botnet rows >20k apart then take the botnet row and previous 20k background rows

Place script in local dir
"""

#Getting all .csv files(local directory) to be feature generated
localFiles = [file for file in os.listdir('.') if file.endswith(".csv")]

#Redistribute the data for each file
for file in localFiles:
    print("Redistributing for file: "+file)
    
    fullDF = pd.read_csv(file)
    fullDF = fullDF.reset_index()

    #Getting index of all Botnet files
    indexBotnet = fullDF[fullDF['Label'].str.contains("Botnet")].index.tolist()


    print("Amount of botnet rows: " + str(len(indexBotnet)))

    finalDF = pd.DataFrame(columns=fullDF.columns)
    allRowsToConcat = []
    if len(indexBotnet)==0:     #No botnet files so just use the full dataframe (all background)
        allRowsToConcat.append(fullDF)

    #Go through each botnet index and get previous 10,100 rows (rolling window will be 10,000)
    #Note: Don't want duplicate rows, so only do this if previous botnet was over 10,000 away, otherwise just continue concatting
    
    for i, index in enumerate(indexBotnet):
        
        print("Copying for botnet row: "+str(i))
        
        #First element
        if i==0:
            if index <= 10002:   #First botnet in first 10,000 rows
                allRowsToConcat.append(fullDF[0:index+1])
            else: #Index over 10,000 so use previous 10,000 rows
                allRowsToConcat.append(fullDF[index-10002:index+1])
      
        else:   #Not first element so check distance to previous botnet row (take 20,000 so we don't skew data for the last 10,000 background rows)
            if (indexBotnet[i]-indexBotnet[i-1]>20002):
                allRowsToConcat.append(fullDF[index-20001:index+1])
            else:
                allRowsToConcat.append(fullDF[indexBotnet[i-1]+1:index+1] )

    print("Concatenating dataframe...")

    finalDF = pd.concat( allRowsToConcat )

    if 'index' in list(finalDF.columns.values):
        finalDF = finalDF.drop(['index'], axis=1)


    #Outputting file
    dataFrameOutDir = Path("ResampledData")
    dataFrameOutDir.mkdir(parents=True, exist_ok=True)
    
    dataframeOutName = file[:-4] + "Resampled.csv"

    finalDF.to_csv(dataFrameOutDir / dataframeOutName, encoding='utf-8', index=False)
    print(dataframeOutName + " Created")

    print("Original total number of rows: "+ str(len(fullDF.index)))
    
    print("New total number of rows: " +   str(len(finalDF.index)))
    print("Total number of normal: " +   str(len(finalDF.index)-len(indexBotnet))  )
    print("Total number of botnet rows: "+  str(len(indexBotnet)))
    print("\n")