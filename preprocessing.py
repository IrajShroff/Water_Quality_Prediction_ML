import numpy as np
import pandas as pd

def read_csv(dataset):

    df = pd.read_csv(dataset)
    return df

## MC 240629
## - add type hinting to function arguments + return values; and, be sure you're returning values
def impute_with_medians(df):
    for column in df.columns[:-1]:
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = df[column].fillna(df[column].median())

def set_variables(df):
    X = df.iloc[:, :-1].values #capital X matrix snd lower case y vector
    y = df.iloc[:, -1].values #values turns it into numpy array
    
    return X,y
