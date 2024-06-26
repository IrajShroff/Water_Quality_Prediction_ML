def Random_Forest_Classifier(X,y):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    rf_clf = RandomForestClassifier(random_state=21)
    rf_clf.fit(X_train, y_train)
    data =[[0,0,5,0,0,0,8.39,54.917862,	0],[5.584087,120,24748.687739,7.544869,325.678363,280.467916,8.399735,27.917862,2.559708]]
    mydata = np.array(data)
    y_pred = rf_clf.predict(mydata)
    print(y_pred)
    #from sklearn.metrics import accuracy_score
    #accuracy_score(y_test,y_pred)

def Logistic_Regression(X,y):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split 
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.pipeline import make_pipeline  # Import make_pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    # Logistic Regression model without scaling for comparison
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)


    # Accuracy 
    accuracy = model.score(X_test, y_test)
    print("Accuracy of Logistic Regression:", accuracy)