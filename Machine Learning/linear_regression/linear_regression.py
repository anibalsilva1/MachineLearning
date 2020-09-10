import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,ticker
import sys




data = pd.read_csv("ex1data1.csv",header=None)


m = data[0].size
nc = len(data.columns)


X_series = data.iloc[:,0:nc-1]
y_series = data.iloc[:,nc-1:nc]

#Insert Series of 1 into X Series

X_series.insert(loc=0,column='ones',value=np.ones((m,1),dtype='int'))

#Transform them into arrays and matrices

X = X_series.values
y = y_series.values

nc_x = len(X_series.columns)

theta = np.zeros((1,nc_x))



def costFunction(X,theta,y):
    
    J = (1/(2*m))*np.sum(np.dot(theta,X.T).T-y)
    
    return J

print(y.shape)
print((np.dot(theta,X.T).shape))


iterations = 1500
alpha = 0.01

def gradDescent(X,y,theta,alpha):
    
    J_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,nc_x))
    
    
    for it in range(iterations):
        
        theta = theta - (alpha / m) * (X.T.dot(np.dot(theta,X.T).T-y).T)
                              
        theta_history[it,:] = theta
        J_history[it] = costFunction(X,theta,y)

    return theta, J_history, theta_history

theta, J_history, theta_history = gradDescent(X,y,theta,alpha)

def normalEquation(X,y):
    
    theta_norm = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)
    
    return theta_norm


theta_norm = normalEquation(X,y)


theta0_sample = np.linspace(-10,10,100)
theta1_sample = np.linspace(-1,4,100)

print(theta)

t = np.zeros((1,nc_x))

J_vals_shape = (len(theta0_sample),len(theta1_sample))

J_vals = np.zeros(J_vals_shape)

for i in range(len(theta0_sample)):
    for j in range(len(theta1_sample)):
        t = np.array([theta0_sample[i],theta1_sample[j]]).reshape(1,2)
        J_vals[i][j] = costFunction(X,t,y)


# plt.scatter(data[0],data[1],marker="x")
# plt.plot(X[:,1], np.dot(theta,X.T).T,color='red')

# plt.xlabel("Population of City in 10,000s")
# plt.ylabel("Profit in 10,000$")
# plt.title("City profit over population")
# plt.show()

print(theta.shape,X.shape)

# plt.plot([i for i in range(iterations)],[J_history[i] for i in range(iterations)])

# fig = plt.figure()
# ax = fig.gca(projection='3d')




# ##### Plots ######

# plot_properties = {
    
#     "cstride": 1,
#     "rstride":1,
#     "cmap":'viridis',
#     "edgecolor":'none'
    
#     }

#  # Plot the surface.
# surf = ax.plot_surface(theta0_sample, theta1_sample, J_vals,**plot_properties)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)


# plt.contour(theta0_sample,theta1_sample,J_vals.T, levels=100)
# plt.scatter(theta[0][0],theta[0][1])
# plt.xlabel("theta0")
# plt.ylabel("theta1")
# plt.show()




