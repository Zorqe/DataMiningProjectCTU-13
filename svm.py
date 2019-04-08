import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import os
import pickle
import numpy as np

#Importing dataframes
test_dataFrame = pd.read_csv('testDataCombinedCaptureNewFeatureGenerated.csv')

#Dropping values that weren't trained on
test_dataFrame = test_dataFrame.drop(['StartTime','SrcAddr','DstAddr'], axis=1)

#Splitting test set from it's class
X_test = test_dataFrame.drop('LabelDisc', axis=1)  
Y_test = test_dataFrame['LabelDisc']

#Stating support vector machine classifier type
svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly')      #Polyclassifier uncomment

#Model doesn't exist so make model and then save it
if not(os.path.exists("svmLinear.pkl")):
#if not(os.path.exists("svmPoly.pkl")):         #Polyclassifier uncomment
    print("\nNo previous model exists, loading training data...")

    train_dataFrame = pd.read_csv('trainDataCombinedCaptureNewFeatureGenerated.csv')
    train_dataFrame = train_dataFrame.drop(['StartTime','SrcAddr','DstAddr'], axis=1)


    #Guide by https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    X = train_dataFrame.drop('LabelDisc', axis=1)  
    Y = train_dataFrame['LabelDisc'] 

    print("Model training...")
    #Training classifier
    svclassifier.fit(X, Y)

    #Save the model
    model_filename = "svmLinear.pkl"
    #model_filename = "svmPoly.pkl"         #Polyclassifier uncomment
    with open(model_filename, 'wb') as file:  
        pickle.dump(svclassifier, file)

#Model exists so load it
else:
    print("Loading model...")
    # Load from file
    #Source(load model): https://stackabuse.com/scikit-learn-save-and-restore-models/
    model_filename = "svmLinear.pkl"  
    #model_filename = "svmPoly.pkl"         #Polyclassifier uncomment
    with open(model_filename, 'rb') as file:  
        svclassifier = pickle.load(file)


#Predicting for the test dataset
print("Predicting classes for test data...")
y_pred = svclassifier.predict(X_test)



#Self classification report:
countMaliciousPredicted = np.count_nonzero(y_pred == 1)
countBackgroundPredicted = np.count_nonzero(y_pred == 0)
totalMalicious = np.count_nonzero(test_dataFrame['LabelDisc'] == 1)

print("The count of malicious predictions are: "  + str( countMaliciousPredicted )  )
print("The count of background predictions are: "  + str( countBackgroundPredicted )  )

print ('\nAccuracy = {:0.2f}%'.format(100.0 * accuracy_score(Y_test, y_pred)))

correctlyClassified = 0
correctlyClassifiedBackground = 0
index = 0
for classification in list(test_dataFrame.LabelDisc):
    if y_pred[index] == classification and classification == 1:
        correctlyClassified += 1
    elif y_pred[index] == classification and classification == 0:
        correctlyClassifiedBackground += 1
    index += 1

print("The number of correctly classified malicious predictions are: " + str(correctlyClassified))

print("The precision on malicious data classification is: {:0.2f}%\n".format( (correctlyClassified/countMaliciousPredicted)*100))
print("The recall on malicious data classification is: {:0.2f}%\n".format( (correctlyClassified/totalMalicious)*100))

print("The precision on benign data classification is: {:0.2f}%\n".format( (correctlyClassifiedBackground/countBackgroundPredicted)*100))


print("Scikit learn Classification report: ")
print(classification_report(Y_test, y_pred))