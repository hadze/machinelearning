import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5,2)), range(5)



#train_test_split(y, shuffle=False)

def Train_Test_Splitter(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test

x_train, x_test, y_train, y_test = Train_Test_Splitter(X,y)

x_train