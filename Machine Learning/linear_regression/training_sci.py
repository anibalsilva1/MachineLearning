import matplotlib.pyplot as plt
import numpy as np


rng = np.random.RandomState(42)

x = 10*rng.rand(50)
y = 2*x - 1 +rng.rand(50)



from sklearn.linear_model import LinearRegression


model = LinearRegression(fit_intercept=True)

X = x[:,np.newaxis]
y = y[:,np.newaxis]



model.fit(X,y) ## Calculates the model parameteres theta_0, theta_1 according with data

x_fit = np.linspace(-1,11)
X_fit = x_fit.reshape((50,1))


y_fit = model.predict(X_fit)

print(y_fit)

# plt.plot(X_fit,y_fit)
# plt.scatter(x,y)
# plt.show()
