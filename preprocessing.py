import numpy as np
import pandas as pd

def read_csv(dataset)-> pd.DataFrame:

    df = pd.read_csv(dataset)
    return df

def impute_with_medians(df)-> pd.DataFrame:
    for column in df.columns[:-1]:
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = df[column].fillna(df[column].median())
    return df

def set_variables(df) -> np.ndarray:
    X = df.iloc[:, :-1].values #capital X matrix snd lower case y vector
    y = df.iloc[:, -1].values #values turns it into numpy array
    
    return X,y
