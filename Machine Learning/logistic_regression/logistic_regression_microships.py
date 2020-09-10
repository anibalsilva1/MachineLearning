import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


df = pd.read_csv("ex2data2.csv", names=['microship1','microship2','approved'])

nc = len(df.columns)
m =len(df.index)

X1 = df.iloc[:,nc-3].values
X2 = df.iloc[:,nc-2].values
y  = df.iloc[:,nc-1].values.reshape((m,1))
X  = df.iloc[:,nc-3:nc-1]

X_pos = df[df.approved == 1]
X_neg = df[df.approved == 0]

#feature matrix for non-linear classification problem

def polinomial(V1,V2,degree):

    res = np.ones(V1.shape[0])

    for i in range(1,degree+1):

        for j in range(0,i+1):

            res = np.column_stack((res,(V1**(i-j))*(V2**j)))


    return res

degree = 6

X_poly = polinomial(X1,X2,degree)
theta_zero = np.zeros((X_poly.shape[1],1))


def sigmoid(z):

    return 1 / (1+np.exp(z))

reg_param=1

def costFunction(theta,X,y):

    z = X.dot(theta.T)
    h = sigmoid(z)
    
    term1 = y.T.dot(np.log(h))
    term2 = (1- y).T.dot(np.log(1 - h))
    J = -np.sum (term1 + term2) / m

    regularization_term = (reg_param/m)*np.sum((theta[1:])**2)

    J_reg = J + regularization_term
    return J_reg

#Find the thetas that minimize costFunction
res = minimize(costFunction, theta_zero, args=(X_poly, y))

theta = res.x

print(theta.shape,theta_zero.shape)

print(costFunction(theta.T,X_poly,y))
print(costFunction(theta_zero.T,X_poly,y))

u = np.linspace(-1,1,118).reshape((118,1))
v = np.linspace(-1,1,118).reshape((118,1))

U,V = np.meshgrid(u,v)
#Flatten the matrix to calculate non-linear features values
U = np.ravel(U)
V = np.ravel(V)

Z = np.zeros(len(u)*len(v))
X_poly2 = polinomial(U, V, degree)

Z = X_poly2.dot(theta.T)
#Revert back to matrix
U = U.reshape(len(u),len(v))
V = V.reshape(len(u),len(v))
Z = Z.reshape(len(u),len(v))



plt.scatter(X_pos.iloc[:,0], X_pos.iloc[:,1] ,marker='x',label='approved')
plt.scatter(X_neg.iloc[:,0],X_neg.iloc[:,1], marker='o', label='not approved')
plt.contour(U,V,Z,levels=[0])


plt.show()

