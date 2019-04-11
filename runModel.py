import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

"""
Input: Processed testing data (local directory) and the trained models
Output Display: Predicted classes with classification reports

Script to test all the models: AdaBoost, Gradient Boost, SVM, and Decision Tree
After predicting the classes for the test data the classification reports are printed
"""

#Load the testing dataframe
print("Loading testing data...")

test_dataFrame = pd.read_csv('finalTestingData.csv')

#Dropping values that weren't trained on
test_dataFrame = test_dataFrame.drop(['StartTime','SrcAddr','DstAddr'], axis=1)

#Splitting up the test dataframe into one with only features, other with classifications
test_dataFrame_Classification = test_dataFrame[['LabelDisc']].copy()
test_dataFrame = test_dataFrame.drop(['LabelDisc'], axis=1)

#Getting vals to use for testing
test_data = test_dataFrame.values
test_features = test_data[0::]

#Testing
modelFileNames = ["adaBoostModel.pkl","LinearSVCModel.pkl"]

model = None
for model_filename in modelFileNames:
    print("\n\nTesting using model "+model_filename+"...")
    with open(model_filename, 'rb') as file:  
        model = pickle.load(file)

    predictionClassification = model.predict(test_features)

    print("Scikit-learn classification report: ")
    print(classification_report(test_dataFrame_Classification.LabelDisc, predictionClassification))  