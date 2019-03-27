import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dataFrameTrain = pd.read_csv('trainDataCombinedCaptureNewFeatureGenerated.csv')
dataFrameTest = pd.read_csv('testDataCombinedCaptureNewFeatureGenerated.csv')


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

print("Dropping cols to not train on")

dataFrameTrain = dataFrameTrain.drop(['StartTime','SrcAddr','DstAddr','SrcAddr_App','SrcDst_Sport_unique'], axis=1)
dataFrameTest = dataFrameTest.drop(['StartTime','SrcAddr','DstAddr','SrcAddr_App','SrcDst_Sport_unique'], axis=1)

print("Normalizing data")

dataFrameTrain = normalizeData(dataFrameTrain)
dataFrameTest = normalizeData(dataFrameTest)

#Predicting using the KNN model
k = 5
print("\nPredicting values with k value: " + str(k))
predictions = knnPredict(k, dataFrameTest, dataFrameTrain)

countMaliciousPredicted = np.count_nonzero(predictions == 15.0)
countBackgroundPredicted = np.count_nonzero(predictions == 0.0)

totalMalicious = np.count_nonzero(dataFrameTest['LabelDisc'] == 1)

print("The count of malicious predictions are: "  + str( countMaliciousPredicted )  )
print("The count of background predictions are: "  + str( countBackgroundPredicted )  )

#Calculating the accuracy
accuracy =  accuracy_score(dataFrameTest['LabelDisc'], predictions) *100
print ("The accuracy is: " + str(accuracy)+"%")

correctlyClassified = 0
correctlyClassifiedBackground = 0
index = 0
for classification in list(dataFrameTest.LabelDisc):
    if predictions[index] == classification and classification == 15.0:
        correctlyClassified += 1
    elif predictions[index] == classification and classification == 0.0:
        correctlyClassifiedBackground += 1
    index += 1

print("The number of correctly classified malicious predictions are: " + str(correctlyClassified))


print("The precision on malicious data classification is: {:0.2f}%\n".format( (correctlyClassified/countMaliciousPredicted)*100))
print("The recall on malicious data classification is: {:0.2f}%\n".format( (correctlyClassified/totalMalicious)*100))

print("The precision on benign data classification is: {:0.2f}%\n".format( (correctlyClassifiedBackground/countBackgroundPredicted)*100))