import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import pickle


#Inspired from: https://www.kaggle.com/treina/titanic-with-adaboost
#---Main---

#An adaboost DT classifier using RandomForests
model = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),
                        algorithm="SAMME",
                        n_estimators=500)

test_dataFrame = pd.read_csv('testDataCombinedCaptureNewFeatureGenerated.csv')

#Dropping values that weren't trained on
test_dataFrame = test_dataFrame.drop(['StartTime','SrcAddr','DstAddr'], axis=1)

#Splitting up the test dataframe into one with only features, other with classifications
test_dataFrame_Classification = test_dataFrame[['LabelDisc']].copy()
test_dataFrame = test_dataFrame.drop(['LabelDisc'], axis=1)

test_data = test_dataFrame.values
test_data_results = test_dataFrame_Classification.values

#Getting vals to use for testing
test_features = test_data[0::]
test_results = test_data_results[0::,0]

#Model doesn't exist so make model and then save it
if not(os.path.exists("adaBoostModel.pkl")):

    print("\nNo previous model exists, loading training data...")

    train_dataFrame = pd.read_csv('trainDataCombinedCaptureNewFeatureGenerated.csv')

    #Drop columns to not train on
    print("Dropping columns not required for training...")

    train_dataFrame = train_dataFrame.drop(['StartTime','SrcAddr','DstAddr'], axis=1)

    train_dataframe_Classification = train_dataFrame[['LabelDisc']].copy()
    train_dataFrame = train_dataFrame.drop(['LabelDisc'], axis=1)

    #-------Training--------
    #Getting the values for the data frame as well as the results (0 normal, 1 malicious)
    train_data = train_dataFrame.values
    train_data_results = train_dataframe_Classification.values

    #Storing the features to use in model as well as the classifications
    train_features = train_data[0::]
    train_result = train_data_results[0::,0]


    print("Model training...")

    #Fit the training data to the Adaboost model
    model = model.fit(train_features, train_result)

    #Save model
    #Source(save model): https://stackabuse.com/scikit-learn-save-and-restore-models/
    model_filename = "adaBoostModel.pkl"
    with open(model_filename, 'wb') as file:  
        pickle.dump(model, file)

#Model exists so load it
else:
    print("Loading model...")
    # Load from file
    #Source(load model): https://stackabuse.com/scikit-learn-save-and-restore-models/
    model_filename = "adaBoostModel.pkl"  
    with open(model_filename, 'rb') as file:  
        model = pickle.load(file)

print("Predicting...")

prediction = model.predict(test_features)


countMaliciousPredicted = np.count_nonzero(prediction == 1)
countBackgroundPredicted = np.count_nonzero(prediction == 0)

totalMalicious = np.count_nonzero(test_dataFrame_Classification['LabelDisc'] == 1)

print("The count of malicious predictions are: "  + str( countMaliciousPredicted )  )
print("The count of background predictions are: "  + str( countBackgroundPredicted )  )

print ('\nAccuracy = {:0.2f}%'.format(100.0 * accuracy_score(test_results, prediction)))

correctlyClassified = 0
correctlyClassifiedBackground = 0
index = 0
for classification in list(test_dataFrame_Classification.LabelDisc):
    if prediction[index] == classification and classification == 1:
        correctlyClassified += 1
    elif prediction[index] == classification and classification == 0:
        correctlyClassifiedBackground += 1
    index += 1

print("The number of correctly classified malicious predictions are: " + str(correctlyClassified))

print("The precision on malicious data classification is: {:0.2f}%\n".format( (correctlyClassified/countMaliciousPredicted)*100))
print("The recall on malicious data classification is: {:0.2f}%\n".format( (correctlyClassified/totalMalicious)*100))

print("The precision on benign data classification is: {:0.2f}%\n".format( (correctlyClassifiedBackground/countBackgroundPredicted)*100))



print("Scikit-learn classification report: ")
print(classification_report(test_dataFrame_Classification.LabelDisc, prediction))  