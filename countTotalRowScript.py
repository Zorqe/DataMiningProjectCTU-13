#Script to output counts of total files
import pandas as pd
import os

"""
This script returns total counts for the rows
Place script in local dir
"""

#Getting all .csv files(local directory) to be feature generated
localFiles = [file for file in os.listdir('.') if file.endswith(".csv")]

totalRows = 0
totalBotnetRows = 0


for file in localFiles:
    print("Counting rows for file: "+file)
    dataFrame = pd.read_csv(file)
    totalBotnetRows += len(dataFrame[dataFrame['Label'].str.contains("Botnet")].index.tolist())
    totalRows += len(dataFrame.index)

print("\n")
print("The total number of rows is: " + str(totalRows))
print("The total number of Botnet rows is: " + str(totalBotnetRows))
print("The total number of Background/Normal rows is: " + str(totalRows - totalBotnetRows))