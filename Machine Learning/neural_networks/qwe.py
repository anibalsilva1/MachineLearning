import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

q = np.random.randint(4,size=(2,2,2))

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels), (test_images,test_labels) = fashion_mnist.load_data()

print(np.argmax(train_images, axis=1))