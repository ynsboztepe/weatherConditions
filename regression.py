import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("maxTemperatureData.csv")

print(df)
print(df.info())

features = ["MinTemp","Precip","ELEV","Latitude","Longitude"]
result = ["MaxTemp"]

X = df[features]
y = df["MaxTemp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

lasso_cv_reg = LassoCV(cv=5)
lasso_cv_reg.fit(X_train, y_train)
y_pred = lasso_cv_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Adj_r2 = 1 - (1-r2_score(y_test, y_pred)) * (len(y_test)-1)/(len(y_test)-X.shape[1]-1)

print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)
print("Adjusted R2 Score: ", Adj_r2)

sns.scatterplot(x=y_test, y=y_pred)
plt.title("y_test lasso CV")
plt.savefig("y_test_lassoCV.png")
plt.close()

print("Alpha:", lasso_cv_reg.alpha_)
print("Alphas", lasso_cv_reg.alphas_)