# pip install lazypredict

from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
from dataEDA import df

diabetes  = df

features = ["MinTemp","Precip","ELEV","Latitude","Longitude"]
result = ["MaxTemp"]

X, y = shuffle(diabetes[features], diabetes[result], random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)