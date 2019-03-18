import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Train dataset to use for classification
#dataFrameTrain = pd.read_csv('trainDataSetFeatureGenerated.csv')
#dataFrameToPredict = pd.read_csv('trainDataSetFeatureGenerated.csv')


#--------------------TEMPORARY----------------------
dataFrame = pd.read_csv('capture20110818ResampledFeatureGenerated.csv')

dataFrameTrain = dataFrame[0:390000]
dataFrameTest = dataFrame[390000:]


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
dataFrameTrain = dataFrameTrain.drop(['StartTime','SrcAddr','DstAddr','State','SrcAddr_App','SrcDst_Sport_unique'], axis=1)
dataFrameTest = dataFrameTest.drop(['StartTime','SrcAddr','DstAddr','State','SrcAddr_App','SrcDst_Sport_unique'], axis=1)

dataFrameTrain = normalizeData(dataFrameTrain)
dataFrameTest = normalizeData(dataFrameTest)

#Predicting using the KNN model
k = 1
print("\nPredicting values with k value: " + str(k))
predictions = knnPredict(k, dataFrameTest, dataFrameTrain)

print("Set of predictions are: " + str(set(predictions)) )
print("The count of malicious predictions are: "  + str( np.count_nonzero(predictions == 15.0) )  )
print("The count of background predictions are: "  + str( np.count_nonzero(predictions == 0.0) )  )

#Calculating the accuracy
accuracy =  accuracy_score(dataFrameTest['LabelDisc'], predictions) *100
print ("The accuracy is: " + str(accuracy)+"%")

correctlyClassified = 0
index = 0
for classification in list(dataFrameTest.LabelDisc):
    if predictions[index] == classification and classification == 15.0:
        correctlyClassified += 1
    index += 1

print("The number of correctly classified malicious predictions are: " + str(correctlyClassified))