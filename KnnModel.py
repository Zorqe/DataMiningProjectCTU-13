import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Train dataset to use for classification
#dataFrameTrain = pd.read_csv('trainDataSetFeatureGenerated.csv')
#dataFrameTrain = pd.read_csv('capture20110815-2FeatureGenerated.csv')
#dataFrameToPredict = pd.read_csv('trainDataSetFeatureGenerated.csv')


#--------------------TEMPORARY----------------------
dataFrame = pd.read_csv('testing100000RowsFeatureGenerated.csv')
dataFrameTrain = dataFrame[0:85000]
dataFrameTest = dataFrame[85000:]


#Mapping taken from tutorial https://www.kaggle.com/parasjindal96/how-to-normalize-dataframe-pandas
#Function to map strings and discretize for discretization
def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data


#Function to normalize all data for similarity calculations is KNN
def normalizeData(dfTrain):
    stringColsToMap = ['Sport','Dport','TotBytesDisc','SrcBytesDisc','SportDisc','DportDisc','Src_TotBytesDisc_mode','Dst_TotBytesDisc_mode']
    
    dfTrain['Sport']=pd.to_numeric(dfTrain['Sport'], errors='coerce')
    dfTrain['Dport']=pd.to_numeric(dfTrain['Dport'], errors='coerce')


    for col in stringColsToMap:
        dfTrain = mapping(dfTrain,col)


    #Normalizing all values between 0 to 15
    dfNormalized = ((dfTrain-dfTrain.min())/(dfTrain.max()-dfTrain.min()))*15
    
    dfNormalized = dfNormalized.fillna(0)
    return dfNormalized




#Function to predict the class of all entries using the KnnClassifier from Sikit-learn
#Returns the list of predictions
def knnPredict(k, dataFrameTest,dataFrameTrain):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(dataFrameTrain.drop(['LabelDisc'],axis=1), dataFrameTrain['LabelDisc'])

    predictions = knn.predict(dataFrameTest.drop(['LabelDisc'],axis=1))
    return predictions
    




#Drop all columns which we won't use for training data, instead use their discretized values
dataFrameTrain = dataFrameTrain.drop(['StartTime','SrcAddr','DstAddr','State','Label','Proto','SrcAddr_App','SrcDst_Sport_unique'], axis=1)
dataFrameTest = dataFrameTest.drop(['StartTime','SrcAddr','DstAddr','State','Label','Proto','SrcAddr_App','SrcDst_Sport_unique'], axis=1)

dataFrameTrain = normalizeData(dataFrameTrain)
dataFrameTest = normalizeData(dataFrameTest)

#Predicting using the KNN model
k = 1
print("\nPredicting values with k value: " + str(k))
predictions = knnPredict(k, dataFrameTest, dataFrameTrain)

print("The predicted discretized labels are: \n"+ str(predictions))

#Calculating the accuracy
accuracy =  accuracy_score(dataFrameTest['LabelDisc'], predictions) *100
print ("The accuracy is: " + str(accuracy)+"%")

print(str(set(predictions)))