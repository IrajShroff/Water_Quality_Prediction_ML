## MC 240629
## - cleaning up and organizing imports
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def split_data(X: np.ndarray, y: np.ndarray, test_size: float) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=21)
    return X_train, X_test, y_train, y_test

def random_forest_classifier(X_train: np.ndarray,y_train: np.ndarray) -> RandomForestClassifier:
    rf_clf = RandomForestClassifier(random_state=21)
    rf_clf.fit(X_train, y_train)
    return rf_clf

def logistic_regression(X_train: np.ndarray,y_train: np.ndarray)-> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def support_vector_machines(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray)-> SVC:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = SVC(kernel = 'rbf', random_state = 0)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return classifier

def scalar_transformation(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train,X_test

def accuracy_of_model(X_test: np.ndarray, y_test: np.ndarray, model)-> float:
    accuracy = model.score(X_test, y_test)
    print("Accuracy of your Model: ", accuracy)
    return accuracy

def k_fold_cross_validation(X: np.ndarray,y: np.ndarray):
    kf = StratifiedKFold(n_splits=5, shuffle = True, random_state=43)
    cnt = 1
    for train_index, test_index in kf.split(X,y):
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set: {len(test_index)}')
        cnt+=1

    score1 = cross_val_score(RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores for each fold are: {score1}')
    print(f'Average score Random Forest Classifier: {"{:.2f}".format(score1.mean())}')

    score2 = cross_val_score(SVC(random_state=42), X, y, cv=kf, scoring="accuracy")

    print(f'Scores for each fold are: {score2}')
    print(f'Average score SVM: {"{:.2f}".format(score2.mean())}')
    
    return score1, score2
