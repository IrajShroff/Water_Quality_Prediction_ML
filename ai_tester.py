from preprocessing import read_csv, impute_with_medians,set_variables
from models import random_forest_classifier, logistic_regression ,support_vector_machines, split_data, accuracy_of_model, k_fold_cross_validation

csv = "./water_potability.csv"


dataFile = read_csv(csv)
impute_with_medians(dataFile)        
X, y = set_variables(dataFile)
X_train, X_test, y_train, y_test = split_data(X,y,0.2)


#model = random_forest_classifier(X_train, y_train)
#model = logistic_regression(X_train, y_train)
model = support_vector_machines(X_train, X_test, y_train, y_test)
accuracy_of_model(X_test, y_test, model)

k_fold_cross_validation(X,y)