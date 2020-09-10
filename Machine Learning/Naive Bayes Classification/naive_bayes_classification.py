from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

iris = sns.load_dataset('iris') 
iris.head()

X_iris = iris.drop('species', axis=1)

y_iris = iris['species']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB() 
model.fit(Xtrain, ytrain) 
y_model = model.predict(Xtest)
