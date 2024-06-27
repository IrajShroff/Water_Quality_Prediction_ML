import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import make_pipeline  # Import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
from sklearn.svm import SVC


def Split_Data(X,y):
    testing_data = float(input("What decimal [up to 1] do you want your testing data to be: "))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_data, random_state=21)
    return X_train, X_test, y_train, y_test

def Random_Forest_Classifier(X_train,y_train):
    
    rf_clf = RandomForestClassifier(random_state=21)
    rf_clf.fit(X_train, y_train)
    data =[[0,0,5,0,0,0,8.39,54.917862,	0],[5.584087,120,24748.687739,7.544869,325.678363,280.467916,8.399735,27.917862,2.559708]]
    mydata = np.array(data)
    y_pred = rf_clf.predict(mydata)
    print(y_pred)
    #from sklearn.metrics import accuracy_score
    #accuracy_score(y_test,y_pred)
    return rf_clf

def Logistic_Regression(X_train, y_train):
   
    # Logistic Regression model without scaling for comparison
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def Support_Vector_Machines(X_train, X_test, y_train, y_test):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = SVC(kernel = 'rbf', random_state = 0)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return classifier

def Accuracy_of_Model(X_test, y_test, model):
    accuracy = model.score(X_test, y_test)
    print("Accuracy of your Model: ", accuracy)

def K_Fold_Cross_Validation(X,y):
    kf = StratifiedKFold(n_splits=5, shuffle = True, random_state=43)
    cnt = 1
    for train_index, test_index in kf.split(X,y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set: {len(test_index)}')
        cnt+=1

    score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores for each fold are: {score}')
    print(f'Average score Random Forest Classifier: {"{:.2f}".format(score.mean())}')

    score = cross_val_score(SVC(random_state=42), X, y, cv=kf, scoring="accuracy")

    print(f'Scores for each fold are: {score}')
    print(f'Average score SVM: {"{:.2f}".format(score.mean())}')
