#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn import preprocessing 


#Deletes row's where the column values are null,nan,nat, or blank
def deleteNullRow(dataFrame, column):
    newDataFrame = dataFrame
    
    #dataframe dropna won't replace empty values only NaN and NaT so convert blank space to NaN then drop
    newDataFrame[column].replace('', np.nan, inplace=True)
    newDataFrame = newDataFrame.dropna(subset=[column])
    return newDataFrame


# From: https://github.com/mgarzon/cybersec/blob/master/MalwareDetection.ipynb
def preprocessData(dataFrame):

    '''
    This function is used to perform
    the necessary operations to 
    convert the raw data into a
    clean data set.
    '''
    
    #Outputting number of rows and column names before preprocessing
    print("----------Before pre-processing-----------")
    print("Number of rows: " + str(len(dataFrame.index)))
    print("The columns are: " + str(list(dataFrame)))
    
    
    
    #dropping columns specified
    listOfFeaturesToDrop = [
    'Dir',
    'sTos',
    'dTos']
    dataFrame = dataFrame.drop(listOfFeaturesToDrop, axis=1)

    #Dropping all null value rows from specified columns
    dataFrame = deleteNullRow(dataFrame,'Sport')
    dataFrame = deleteNullRow(dataFrame,'SrcAddr')
    dataFrame = deleteNullRow(dataFrame,'Dport')
    dataFrame = deleteNullRow(dataFrame,'DstAddr')
    
    
    # TODO
    #dp.convertColumnToTimeStamp(dataFrame,'StartTime') # ?? already a timestamp
    
    
    #Outputting number of rows and column names after preprocessing
    print("\n----------After pre-processing-----------")
    print("Number of rows: " + str(len(dataFrame.index)))
    print("The columns are: " + str(list(dataFrame)))
    return dataFrame


#Function to perform discretization on the data
def discretizeData(dataFrame):
    
    dfNew = dataFrame
    
    # Binning technique from
    # https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
    quantile_list = [0, .25, .5, .75, 1.] # Change the quantile_list for more or less accuracy
    
    dfNew['TotBytesDisc'] = ""
    dfNew['SrcBytesDisc'] = ""
    dfNew['TotBytesDisc'] = pd.qcut(dataFrame['TotBytes'], quantile_list)
    dfNew['SrcBytesDisc'] = pd.qcut(dataFrame['SrcBytes'], quantile_list)
    
    # Bin Src/Dest port
    # According to 0-1023(WELLKNOWN_PORTNUMBER)
    #              1024-49151(REGISTERED_PORTNUMBER)
    #              49152-65535(DYNAMIC_PORTNUMBER)
    Sport = dataFrame['Sport']#[0x0303].astype('int64')
    Sport = Sport.apply(lambda x: int(x, 16) if x[0] == '0' and x[1] == 'x' else int(x, 10)) # TODO, there has to be better way
    dfNew['SportDisc'] = ""
    dfNew['SportDisc'] = pd.cut(Sport, [0, 1023, 49151, 65535])
    
    Dport = dataFrame['Dport']#[0x0303].astype('int64')
    Dport = Dport.apply(lambda x: int(x, 16) if x[0] == '0' and x[1] == 'x' else int(x, 10))
    dfNew['DportDisc'] = ""
    dfNew['DportDisc'] = pd.cut(Dport, [0, 1023, 49151, 65535])

    
    #LabelEncoder for unique values for Proto column and stored as column ProtoDisc
    le = preprocessing.LabelEncoder()
    le.fit(dfNew.Proto.unique())
    dfNew["ProtoDisc"] = ""
    dfNew.ProtoDisc = le.transform(dfNew.Proto)
    
    
    #Encoding "label" column to "labelDisc"
    #0 = Background/Normal             1=Botnet
    dfNew["LabelDisc"] = ""
    dfNew['LabelDisc'] = dfNew['Label']
    dfNew['LabelDisc'] = dfNew.LabelDisc.str.replace(r'(^.*Background.*$)', '0')
    dfNew['LabelDisc'] = dfNew.LabelDisc.str.replace(r'(^.*Normal.*$)', '0')
    dfNew['LabelDisc'] = dfNew.LabelDisc.str.replace(r'(^.*Botnet.*$)', '1')
    
    
    return dfNew
    



#helper function to count the distinct values of second column
#where SRCaddr's match in rolling window of size windowSize
def countDistinctMatchingForSrcAddr(sliceDF):
    SrcAddr = sliceDF["SrcAddr"].iloc[-1]     #SrcAddr of the rolling window to calculate for
    DstAddr = sliceDF["DstAddr"].iloc[-1]
    
    returnData = pd.DataFrame()
    
    srcAddrRows = sliceDF[sliceDF.SrcAddr == SrcAddr]
    destAddrRows = sliceDF[sliceDF.DstAddr == DstAddr]
    srcAndDestRows = srcAddrRows[srcAddrRows.DstAddr == DstAddr]
    
    # SrcAddr statistics
    returnData["SrcAddr_App"] = [srcAddrRows.shape[0]]   #counting total SRCAddr matches
    returnData["Src_Dport_unique"] =  srcAddrRows.Dport.nunique() #only counting distinct dports by using set
    returnData["Src_DstAddr_unique"] =  srcAddrRows.DstAddr.nunique()
    returnData["Src_Sport_unique"] =  srcAddrRows.Sport.nunique()
    returnData["Src_TotPkts_mean"] = srcAddrRows.TotPkts.mean()
    returnData["Src_TotBytesDisc_mode"] = srcAddrRows.TotBytesDisc.mode() # not quite mean but close enough
    
    # DstAddr statistics
    returnData["DstAddr_App"] = [destAddrRows.shape[0]]   #counting total SRCAddr matches
    returnData["Dst_Dport_unique"] =  destAddrRows.Dport.nunique()
    returnData["Dst_SrcAddr_unique"] =  destAddrRows.SrcAddr.nunique()
    returnData["Dst_Sport_unique"] =  destAddrRows.Sport.nunique()
    returnData["Dst_TotPkts_mean"] = destAddrRows.TotPkts.mean()
    returnData["Dst_TotBytesDisc_mode"] = destAddrRows.TotBytesDisc.mode() # not quite mean but close enough
    
    # Src+Dstaddr statistics
    returnData["SrcDst_Sport_unique"] =  srcAndDestRows.Sport.nunique()
    returnData["SrcDst_Dport_unique"] =  srcAndDestRows.Dport.nunique()
    
    return returnData



#Function to generate connection based features for the source address
def generateSrcAddrFeaturesConnectionBased(dataFrame, windowSize):
    
    dfNew = dataFrame
    
    #How many times the SRCADDRESS has appeared within the last X netflows (SrcAddr_Dis)
    #For any of the flow records that SRCADDRESS has appeared within the last X netflows, count the distinct destination ports (Src_Dist_Des_Port) 
    #For any of the flow records that SRCADDRESS has appeared within the last X netflows, count the distinct destination addresses (Src_Dist_Des_Addr)
    #For any of the flow records that SRCADDRESS has appeared within the last X netflows, count the distinct source ports (Src_Dist_Src_Port)
    #For any of the flow records that SRCADDRESS AND DSTADDRESS has appeared within the last X netflows, count the distinct source ports   
    #For any of the flow records that SRCADDRESS AND DSTADDRESS has appeared within the last X netflows, count the distinct destinations ports
    
    #For any of the flow records that SRCADDRESS has appeared within the last X netflows, average the packets
    #For any of the flow records that SRCADDRESS has appeared within the last X netflows, average the bytes
    
    additionalCol = None
    for i in range(windowSize - 1, len(dfNew)+1):
        #Feature generation feedback every 10000 generated rows
        if (i%10000 == 0):
            print(i)

        window = dfNew[i - (windowSize-1):i+1]
        
        slice_df = countDistinctMatchingForSrcAddr(window)
        
        if i == windowSize - 1:
            additionalCol = slice_df
        else:
            additionalCol = pd.concat([additionalCol, slice_df])
        
        #dfNew.loc[i, :] = slice_df
    
    # Set the right index
    additionalCol = additionalCol.reset_index()
    additionalCol.index += windowSize - 1
    
    dfNew = dfNew.join(additionalCol).drop(columns=['index'])
    
    
    #Dropping beginning rows of size: windowsize since all the generated features are null
    dfNew = deleteNullRow(dfNew, 'SrcAddr_App')
    
    return dfNew
    



#Have all files to clean,discretize,featuregenerate in local directory
#Getting all .csv files(local directory) to be feature generated
localFiles = [file for file in os.listdir('.') if file.endswith(".csv")]

#Perform preprocessing, discretizing, and feature generation on each file seperately
for file in localFiles:
    print("Reading file: "+file)
    dataFrame = pd.read_csv(file)

    print("Preprocessing file: "+file)
    dataFrame = preprocessData(dataFrame)

    print("Discretizing file: "+file)
    dataFrame = discretizeData(dataFrame)

    print("Feature generating file: "+file)
    #Window size 10,000
    dataFrame = generateSrcAddrFeaturesConnectionBased(dataFrame,10000)
    dataframeOutName = file[:-4] + "FeatureGenerated.csv"
    dataFrame.to_csv(dataframeOutName, encoding='utf-8', index=False)
    print(dataframeOutName + " Created \n\n\n")