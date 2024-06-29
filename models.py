## MC 240629
## - cleaning up and organizing imports
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

## MC 240629
## - functions should be lowercase snake_case
## - include type hinting on function arguments and return values
## - better practice to make `test_size` an argument instead of asking the user through input()
def split_data(X: np.ndarray, y: np.ndarray, test_size: float) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=21)
    return X_train, X_test, y_train, y_test

## MC 240629
## - add type hinting to function arguments and return values for all functions here
## - rename functions to be lowercase snake_case
def Random_Forest_Classifier(X_train,y_train):
    
    rf_clf = RandomForestClassifier(random_state=21)
    rf_clf.fit(X_train, y_train)

    ## MC 240629
    ## - this is a test case, so it shouldn't be in the function
    ## - if you want to test the model, you should write a separate test function
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

    ## MC 240629
    ## - worth making scaler transformation a separate function, so could be reused
    ## - also makes for consistency between these model functions
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = SVC(kernel = 'rbf', random_state = 0)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return classifier

## MC 240629
## - good practice to have functions return values instead of printing them, can always print the returned value
def Accuracy_of_Model(X_test, y_test, model):
    accuracy = model.score(X_test, y_test)
    print("Accuracy of your Model: ", accuracy)

def K_Fold_Cross_Validation(X,y):
    kf = StratifiedKFold(n_splits=5, shuffle = True, random_state=43)
    cnt = 1
    for train_index, test_index in kf.split(X,y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set: {len(test_index)}')
        cnt+=1

    score = cross_val_score(RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores for each fold are: {score}')
    print(f'Average score Random Forest Classifier: {"{:.2f}".format(score.mean())}')

    score = cross_val_score(SVC(random_state=42), X, y, cv=kf, scoring="accuracy")

    print(f'Scores for each fold are: {score}')
    print(f'Average score SVM: {"{:.2f}".format(score.mean())}')
