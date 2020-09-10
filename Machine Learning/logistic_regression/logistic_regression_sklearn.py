import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("ex2data1.csv", names=['exam1','exam2','approved'])

nc = len(df.columns)
m =len(df.index)

X1 = df.iloc[:,nc-3].values.reshape((100,1))
X2 = df.iloc[:,nc-2].values
y  = df.iloc[:,nc-1].values
X  = df.iloc[:,nc-3:nc-1]

X_pos = df[df.approved == 1]
X_neg = df[df.approved == 0]


model = LogisticRegression()

model.fit(X,y)

thetas = np.concatenate((model.intercept_,model.coef_[0][0]),axis=None).reshape((1,2))

ones_matrix = np.ones((m,1)).reshape((100,1))


X_plot=np.concatenate((ones_matrix,X1),axis=1)
linear_function = np.dot(-thetas,X_plot.T).T/model.coef_[0,1]

plt.scatter(X_pos.iloc[:,0],X_pos.iloc[:,1],marker='x',label='approved')
plt.scatter(X_neg.iloc[:,0],X_neg.iloc[:,1],marker='o',label='not approved')
plt.plot(X1,linear_function,color='red')
plt.legend()
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
plt.show()


