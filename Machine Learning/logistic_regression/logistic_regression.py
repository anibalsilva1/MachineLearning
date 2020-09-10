import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ex2data1.csv", names=['exam1','exam2','approved'])

nc = len(df.columns)
m =len(df.index)

X1 = df.iloc[:,nc-3].values
X2 = df.iloc[:,nc-2].values
y  = df.iloc[:,nc-1].values.reshape(100,1)
X  = df.iloc[:,nc-3:nc-1]

X_pos = df[df.approved == 1]
X_neg = df[df.approved == 0]

X.insert(loc=0,column='ones',value=np.ones((m,1),dtype='int'))

theta = np.zeros((nc,1))

def sigmoid(X,theta):

    s = 1 / (1+np.exp(-np.dot(X,theta)))

    return s



s = sigmoid(X,theta)

print(np.dot(X,theta).shape)
#sigmoid(X,theta) = sigmoid(X,theta)


def costFunction(X,theta,y):

    #J = (1/m)*np.sum(-np.log(sigmoid(X,theta)).dot(y)-np.log(1-sigmoid(X,theta)).dot((1-y)))
    J = -(1/m)*np.sum(np.log(sigmoid(X,theta)).T.dot(y)+np.log(1-sigmoid(X,theta)).T.dot(1-y))
    #J = (1/m)*np.sum(-y*np.log(sigmoid(X,theta))-(1-y)*np.log(1-sigmoid(X,theta)))
    #J = -(1/m)*np.sum(y*np.log(sigmoid(X,theta))+(1-y)*(np.log(1-sigmoid(X,theta))))
    

    return J

J = costFunction(X,theta,y)

alpha=0.0001
iterations=50




print(costFunction(X,np.array([-24,0.2,0.2]),y))


def gradDescent(X,y,theta,alpha):
    
    J_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,nc))
    
    
    for it in range(iterations):
        
        theta = theta - (alpha / m) * (np.dot(X.T,sigmoid(X,theta)-y))
                              
        theta_history[it,:] = theta.T
        J_history[it] = costFunction(X,theta,y)

    return theta, J_history, theta_history

theta, J_history, theta_history = gradDescent(X,y,theta,alpha)

# plt.plot([i for i in range(iterations)],[J_history[i] for i in range(iterations)])
# plt.show()
# print((np.dot(X.T,sigmoid(X,theta)-y)).shape,theta.shape)
print(theta[1][0])
#print(sigmoid(np.array([1,45,85]),theta))

# print(sigmoid(np.array([1,45,85]).reshape(1,3),theta))
#fprintf(' -25.161\n 0.206\n 0.201\n');

print(X1)

#print(np.dot(theta,np.array([1,45,85]).reshape(1,3).T))
# plt.scatter(X_pos.iloc[:,0],X_pos.iloc[:,1],marker='x',label='approved')
# plt.scatter(X_neg.iloc[:,0],X_neg.iloc[:,1],marker='o',label='not approved')
# plt.plot(X1,(-(25.161+0.206*X1)/(0.201)))
# plt.xlabel("exame1")
# plt.ylabel("exame2")
# plt.legend()
# plt.show()