import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


x_train_df = pd.read_csv('X_table.csv',header=None)
y_train_df = pd.read_csv('y_table.csv',header=None)

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels), (test_images,test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#To convert pixels to grey scale

train_images = train_images / 255.0
test_images = test_images / 255.0


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

'''
Build the model

Building the neural network requires configuring the layers of the model, then compiling the model.

Set up the layers

The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. 

Hopefully, these representations are meaningful for the problem at hand.

Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.
'''


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Compress to an 28x28 numpy array, each pixel is a feature
    keras.layers.Dense(128, activation='relu'), # Creates activations of a densily-connected hidden layer with ReLU model
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


predictions = probability_model.predict(test_images)

# failed_to_predict_images = np.zeros()
# failed_to_predict_labels = np.zeros()
test_label_shape = test_labels.shape
predicted = np.zeros(test_label_shape)
predicted = np.argmax(predictions)
print(np.argmax(predictions))

count=0
for i in range(predictions.shape[0]):
    if np.argmax(predictions[i]) == test_labels[i]:
        pass
    else:
        count=count+1
        failed_to_predict_images = test_images[i:i-1]
        failed_to_predict_labels = test_labels[i]
        #print(test_images[i])
print(count)
print(failed_to_predict_labels.shape,failed_to_predict_images.shape)

## VER AGORA AS IMAGENS QUE FALHARAM AO PREVER        