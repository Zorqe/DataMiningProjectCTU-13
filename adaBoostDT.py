import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#Inspired from: https://www.kaggle.com/treina/titanic-with-adaboost

#Function to normalize all data for similarity calculations is KNN
def normalizeData(dfTrain):
    stringColsToMap = ['TotBytesDisc','SrcBytesDisc','SportDisc','DportDisc','Src_TotBytesDisc_mode','Dst_TotBytesDisc_mode']


    dfNormalized = dfTrain
    dfNormalized['Sport']=pd.to_numeric(dfNormalized['Sport'], errors='coerce')
    dfNormalized['Dport']=pd.to_numeric(dfNormalized['Dport'], errors='coerce')

    for col in stringColsToMap:
        LE = LabelEncoder()
        dfNormalized[col] = LE.fit_transform(dfNormalized[col])
    
    dfNormalized = dfNormalized.fillna(0)

    return dfNormalized





#---Main---


dataFrame = pd.read_csv('testing100000RowsFeatureGenerated.csv',encoding="utf-8")

train_dataFrame = dataFrame[0:85000]
test_dataFrame = dataFrame[85000:]

#Drop columns to not train on
train_dataFrame = train_dataFrame.drop(['StartTime','SrcAddr','DstAddr','State','Label','Proto'], axis=1)
test_dataFrame = test_dataFrame.drop(['StartTime','SrcAddr','DstAddr','State','Label','Proto'], axis=1)

train_dataFrame = normalizeData(train_dataFrame)
test_dataFrame = normalizeData(test_dataFrame)

train_dataframe_Features = train_dataFrame[['LabelDisc']].copy()
train_dataFrame = train_dataFrame.drop(['LabelDisc'], axis=1)

test_dataFrame_Features = test_dataFrame[['LabelDisc']].copy()
test_dataFrame = test_dataFrame.drop(['LabelDisc'], axis=1)

#-------Training--------
#Getting the values for the data frame as well as the results (0 normal, 1 malicious)
train_data = train_dataFrame.values
train_data_results = train_dataframe_Features.values

test_data = test_dataFrame.values
test_data_results = test_dataFrame_Features.values

#Storing the features to use in model as well as the classifications
train_features = train_data[0::]
train_result = train_data_results[0::,0]

test_features = test_data[0::]
test_results = test_data_results[0::,0]

print("Model training...")

#An adaboost DT classifier using RandomForests
model = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),
                         algorithm="SAMME",
                         n_estimators=500)

#Fit the training data to the Adaboost model
model = model.fit(train_features, train_result)

print("Predicting...")

prediction = model.predict(test_features)
print(str(set(prediction)))
print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(test_results, prediction)))