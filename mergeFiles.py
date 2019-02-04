#Script to merge all 13 .binetflow csv files
#NOTE: Must have all 13 .csv files locally in same folder as script

import os, csv

#Getting all .csv files(local directory) to be merged into one file
localFiles = [file for file in os.listdir('.') if file.endswith(".csv")]

outFile = open("combinedCapture.csv","a")

count = 0 #count of rows in new csv

#Copy all files
for fileName in localFiles:
    fileToCopy = open(fileName, "r")
    print("Merging: " + fileName)

    #Include the column header only with the first file
    if not(fileName == localFiles[0]):
        fileToCopy.__next__()

    for line in fileToCopy:     #write lines
        if (line[-1] != '\n'):      #prevent run on rows by error
            line += '\n'
        outFile.write(line)
        count += 1
    
    fileToCopy.close()

outFile.close() #Close combined output file

print ("The number of rows(including header) for the final merged file 'combinedCapture.csv' is: " + str(count))