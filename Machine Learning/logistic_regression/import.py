import numpy as np

x = np.linspace(1,100,50).reshape(50,1)
y = np.linspace(-1,-100,50).reshape(1,50)




print(np.sum(y.dot(x)))

print(x.shape,y.shape)