import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



data = pd.read_csv("ex1data1.csv",header=None)

m = data[0].size
nc = len(data.columns)

X_series = data.iloc[:,0:nc-1]
y_series = data.iloc[:,nc-1:nc]

X = X_series.values
y = y_series.values

model = LinearRegression(fit_intercept=True)

model.fit(X,y)


yfit = model.predict(X)
Xfit = model.predict(y)
print(model.score(X, y))



# plt.scatter(X,y)
# #plt.plot(Xfit,y,color='red')
# #plt.plot(X,yfit,color='black')
# plt.plot(Xfit,yfit,color='yellow')

# plt.show()

#print(model.intercept_,model.coef_)