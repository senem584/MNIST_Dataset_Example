import tensorflow as tf
import keras 
from keras import Sequential 
from keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score

# uploading the MNIST dataset using the keras API
# using load_data() splits the dataset into 2 parts: x_train and y_train
# x_train (images) and y_train (labels) are images of handwritten digits and their labels 
# x_test (images) and y_test (labels) are images and labels of handwritten digits for testing the model
(X_train, y_train), (X_test,y_test) = keras.datasets.mnist.load_data()

# checking the dataset to make sure i understand
# 28x28 pixels in the X sets
# 60k samples in training and 10k samples in testing
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# data preprocessing
# we are training and testing with images, so it is important to normalize the pixel values
# pixel values range from 0-255, so we normalize to 0-1
# normalizing allows our model to learn on a smaller scale, so it does not weigh large pixel values as much larger
# x_train and x_test are the only sets with images (the y sets are labels, so they are in an array format)
X_train, X_test = X_train / 255.0, X_test / 255.0

# building the neural network model 
# Sequantial() is used for building neural networks by stacking layers linearly and sequentially
# we can use Sequential() because we have one input and one output
model = Sequential() 
model.add(Flatten(input_shape=(28,28))) # converting the 2D image into a 1D array
# Dense layer = a layer with neurons that are connected to the outputs of the previous layer
# activation function (we are using ReLU) decides what can pass forward
# ReLU basically says if it's positive, keep it. if it is negative, don't. this helps models learn patterns
model.add(Dense(128, activation = 'relu')) # 128 neurons looks at input data in learns to detect a pattern
model. add(Dense(32, activation = 'relu')) # 128 features is decreased to 32 new features that is the most important information
model. add(Dense(11, activation = 'relu')) # producing 11 outputs 
model.summary() 

# compiling and training the model
# epochs = iterations: 1 epoch = 1 complete pass through training dataset 
# through each epoch, weights and biases are adjusted
# calitation_split: prevents overfitting. validates performance on a portion of data
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)

# model evaluation 
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
accuracy_score(y_test,y_pred)
