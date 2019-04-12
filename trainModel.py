import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn import tree

"""
Input: Processed training data (local directory)
Output: All trained machine learning models

Script to train all the models: AdaBoost, Gradient Boost, SVM, and Decision Tree
Outputs the saved models in a pickle formate to be used by runModel.py
"""

#Load the training dataframe
print("Loading training data...")
train_dataFrame = pd.read_csv('finalTrainingData.csv')

#Drop columns to not train on
print("Dropping columns not required for training...")
train_dataFrame = train_dataFrame.drop(['StartTime','SrcAddr','DstAddr'], axis=1)

#Training classes and data seperated
train_dataframe_Classification = train_dataFrame[['LabelDisc']].copy()
train_dataFrame = train_dataFrame.drop(['LabelDisc'], axis=1)

#Getting the values for the classes and the training data
train_data = train_dataFrame.values
train_data_labels = train_dataframe_Classification.values

#Storing the features to use in model as well as the classifications
train_features = train_data[0::]
train_labels = train_data_labels[0::,0]


#Creating all models to be trained on (except KNN)
models = [
        ["adaBoostModel", AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000), algorithm="SAMME", n_estimators=500)],
        ["LinearSVCModel", LinearSVC(max_iter=4000)],
        ["CART", tree.DecisionTreeClassifier()]
        ]

#Training on all models and saving them locally
for modelToTrain in models:
    modelName = modelToTrain[0]
    model = modelToTrain[1]
    print("\n\nModel training for model " + modelName + "...")
    
    #Fit the training data to the model
    model = model.fit(train_features, train_labels)

    print("Saving model "+ modelName + "...")
    model_filename = modelName + ".pkl"
    with open(model_filename, 'wb') as file:  
        pickle.dump(model, file)