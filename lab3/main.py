#!/usr/bin/python
import time

print("Starting...")
start = time.time()

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from sklearn import preprocessing
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

end = time.time()
print("Took", end-start, "seconds to import")

def do_norm(df):
    df_mean = df.mean()
    df_std = df.std()
    df_norm = (df - df_mean) / df_std

    return df_norm.fillna(0, axis=1)

def create_model_simple(my_learning_rate, feature_layer):
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add the layer containing the feature columns to the model.
  model.add(feature_layer)

  # Add one linear layer to the model to yield a simple linear regressor.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Construct the layers into a model that TensorFlow can execute.
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

  return model           

def create_model_deep(my_learning_rate, my_feature_layer):
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add the layer containing the feature columns to the model.
  model.add(my_feature_layer)

  # Describe the topography of the model by calling the tf.keras.layers.Dense
  # method once for each layer. We've specified the following arguments:
  #   * units specifies the number of nodes in this layer.
  #   * activation specifies the activation function (Rectified Linear Unit).
  #   * name is just a string that can be useful when debugging.

  # Define the first hidden layer with 20 nodes.   
  model.add(tf.keras.layers.Dense(units=50, 
                                  activation='relu', 
                                  name='Hidden1'))
  
  # Define the second hidden layer with 12 nodes. 
  model.add(tf.keras.layers.Dense(units=25, 
                                  activation='relu', 
                                  name='Hidden2'))
  
  # Define the output layer.
  model.add(tf.keras.layers.Dense(units=1,  
                                  name='Output'))                              
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="mse",
                loss_weights=0.2,
                metrics=["accuracy"])

  return model



def train_model_simple(model, dataset, epochs, batch_size, label_name):
  """Feed a dataset into the model in order to train it."""

  # Split the dataset into features and label.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True)

  # Get details that will be useful for plotting the loss curve.
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  rmse = hist["mean_squared_error"]

  return epochs, rmse

def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.show()

def plot_the_acc_curve(epochs, acc):
  """Plot a curve of acc vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")

  plt.plot(epochs, acc, label="Gain in accuracy")
  plt.legend()
  plt.ylim([acc.min()*0.95, acc.max() * 1.03])
  plt.show()

def train_model_deep(model, dataset, epochs, label_name,
                batch_size=None):
  """Train the model by feeding it data."""

  # Split the dataset into features and label.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True) 

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  hist = pd.DataFrame(history.history)
  mse = hist["loss"]
  acc = hist["accuracy"]

  return epochs, mse, acc

df = pd.read_csv("./Star3642_balanced.csv")

# shuffle data
#df = df.reindex(np.random.permutation(df.index))

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=10)

train_df = pd.DataFrame(x_train).join(y_train)
test_df = pd.DataFrame(x_test).join(y_test)

# Vmag,Plx,e_Plx,B-V,SpType,Amag,TargetClass
le = preprocessing.LabelEncoder()
all_unique = np.unique(np.concatenate((train_df["SpType"].unique(), test_df["SpType"].unique())))
le.fit(all_unique)
train_df["SpType"] = le.transform(train_df["SpType"])
test_df["SpType"] = le.transform(test_df["SpType"])

feature_columns = []

for i in train_df.columns[:-1]:
    feature_columns.append(tf.feature_column.numeric_column(i))

print(feature_columns)
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 50
batch_size = 1000

# Specify the label
label_name = "TargetClass"
#label_name = "median_house_value"

# Establish the model's topography.
my_model = create_model_simple(learning_rate, my_feature_layer)

# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs, mse = train_model_simple(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, mse)

# After building a model against the training set, test that model
# against the test set.
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)

epochs = 50
batch_size = 1000

# Establish the model's topography.
my_model = create_model_deep(learning_rate, my_feature_layer)

# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs, mse, acc = train_model_deep(my_model, train_df, epochs, 
                          label_name, batch_size)
plot_the_loss_curve(epochs, mse)
plot_the_acc_curve(epochs, acc)

# After building a model against the training set, test that model
# against the test set.
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)

print("Done!")
