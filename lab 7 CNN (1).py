#!/usr/bin/env python
# coding: utf-8

# convolutional Neural Network
# 
# week-7

# **CIFAR dataset:**
# 
# The CIFAR dataset is a popular benchmark dataset in the field of computer vision and machine learning. It is a collection of labeled images used to train and test machine learning models, particularly for image classification tasks. The dataset is widely used because it represents a real-world problem and introduces several challenges that students should be aware of.
# 
# **Classification Problem:**
# 
# The classification problem of the CIFAR dataset is to correctly categorize images into predefined classes. For CIFAR-10, there are ten distinct classes, and for CIFAR-100, there are one hundred classes. Each image belongs to one of these classes, and the goal is to design a model that can accurately assign the correct class label to each image.
# 
# 
# 
# 

# In[3]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split


# In[4]:


# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[5]:


print('X_train size: ',X_train.shape)
print('y_train size: ',y_train.shape)
print('X_test size:  ',X_test.shape)
print('y_test size:  ',y_test.shape)


# In[6]:


# Class names in the CIFAR-10 dataset

classes=['aircraft', 'car', 'bird', 'cat', 'deer',
 'dog', 'frog', 'hours', 'ship', 'truck']

classes


# In[7]:


# Display a few images before model development, Hint - plt.title(f"Label: {y_train[i].argmax()}") will give you True Label.
#
#
# Your code to display a few images before the model development.
#
#
for i in range(5):
  plt.figure(figsize=(1,1))
  plt.imshow(X_train[i])
  plt.title(f"Label: {y_train[i][0]}")
plt.show


# In[8]:


# Preprocess the data
X_train = X_train.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
X_test = X_test.astype('float32') / 255.0
y_test = to_categorical(y_test, 10)


# In[9]:


# Create an MLP model using Keras

mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=X_train.shape[1:]))
mlp_model.add(Dense(512, activation='relu'))
mlp_model.add(Dense(256, activation='relu'))
mlp_model.add(Dense(10, activation='softmax'))

# print the MLP model

print(mlp_model.summary())


# In[10]:


mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[11]:


# Train and evaluate the MLP classifier

history = mlp_model.fit(X_train, y_train, epochs=10, verbose=1)


# In[12]:


# Print the training process graph and result

history_dict = history.history
plt.style.use('seaborn-darkgrid')

acc_values = history_dict['accuracy']
epochs = range(1, len(acc_values) + 1)

plt.figure(num=1, figsize=(15,7))
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[13]:


mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test, verbose=0)
print(f"MLP Test Accuracy: {mlp_accuracy * 100:.2f}%")


# In[14]:


# Create a CNN model using Keras

cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(Flatten())

cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))

# print the CNN model

print(cnn_model.summary())


# In[15]:



# compilation

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


# Train and evaluate the CNN classifier

history2 = cnn_model.fit(X_train, y_train, epochs=10, verbose = 1)


# In[17]:


cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test Accuracy: {cnn_accuracy * 100:.2f}%")


# In[18]:


# Display a few images after CNN model development
for i in range(5):
    plt.figure(figsize=(1, 1))
    plt.imshow(X_train[i])
    plt.title(f"Label: {y_train[i].argmax()}")
    plt.axis('off')
    plt.show()


# Based on the aforementioned code, perform the following activity:
# 
# 
# 1.   Using the provided code, create one distinct CNN models with architectures different from the one provided.
# 
# 
# 2.  Train the model with three different optimizers selected from the options available at https://keras.io/api/optimizers/.
# 
# 3.  Measure the time it takes to train each model with its respective optimizer by importing the 'time' module and recording the start and end times using 'time.time()'.
# 
# 4. Include this bar plot in your lab logbook to compare training times with different optimizers.
# 
# 5. Strive to fine-tune the model parameters to achieve higher accuracy, ideally surpassing 90%. Document in your lab logbook the highest accuracy you have achieved through this fine-tuning process.

# In[19]:


# Your code to do the above activity.
# Hint - import time
#
#
import time
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import to_categorical


# In[20]:


# Your code to do the above activity.
# Hint - Store training time and accuracy in lists
#
#

training_times = []
histories = []
accuracies=[]


# In[21]:


# Your code to do the above activity.
# Hint - Put name of optimizers
#
#
optimizers = [SGD(), Adam(), Adadelta()]


# In[22]:


# Your code to do the above activity.
# Hint - import time and use time.time() just before and after training to record start and end time of the training. Difference will provide the training time.
#
#
#

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


# In[23]:


# Your code to do the above activity.
# Hint - use loops for use different optimizers

for optimizer in optimizers:
    model = create_cnn_model()  # Create a new model for each optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), verbose=1)
    end_time = time.time()
    histories.append(history)
    training_times.append(end_time - start_time)
    accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(accuracy)


# In[24]:


# Your code to do the above activity.
# Generate a bar plot to compare training times
#
#
optimizer_names = ["SGD", "Adam", "Adadelta"]
plt.bar(optimizer_names, training_times)
plt.ylabel('Training Time (seconds)')
plt.xlabel('Optimizer')
plt.title('Training Time with Different Optimizers')
plt.show()


# Lab Logbook Requirements:
# 
# 1. Record the bar graph depicting the comparison of training times.
# 2. Record the final accuracy achieved.
# 

# In[33]:


training_accuracies=[]
for i, optimizer in enumerate(optimizers):
    training_accuracies.append(accuracy)
   
    print(f"{optimizer} Optimizer - Test Accuracy: {accuracy * 100:.2f}%")


# In[34]:


max_index = np.argmax(training_accuracies)
print(f"The highest achieved accuracy is {training_accuracies[max_index] * 100:.2f}% with the {optimizers[max_index]} Optimizer.")


# In[ ]:





# 
