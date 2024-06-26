def preprocess(dataset): 
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset)

    for column in df.columns[:-1]:
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = df[column].fillna(df[column].median())

    X = df.iloc[:, :-1].values #capital X matrix snd lower case y vector
    y = df.iloc[:, -1].values #values turns it into numpy array
    
    return X,y