Python 3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 

import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#/ Everything is heavily referenced from tensorflow.org.

print(tf.__version__)

training_set = pd.read_csv("StudentsPerformance.csv")

training_set.head()
shalow_training_set = training_set.copy()
shalow_training_set.isna().sum()
shalow_training_set = pd.get_dummies(shalow_training_set)
shalow_training_set.head()
shalow_training_set.describe().T
## reference from https://www.tensorflow.org/tutorials/keras/regression
# train set 70
train_set = shalow_training_set.sample(frac=0.7, random_state=0)
new_set = shalow_training_set.drop(train_set.index)

# test set 15
test_set = new_set.sample(frac=0.5, random_state=0)

# validation set 15
validation_set = new_set.drop(test_set.index)

# training/test/validation features (copy of train/test/validation sets)
train_feas = train_set.copy()
test_feas = test_set.copy()
validation_feas = validation_set.copy()

# scores separated sets 70/15/15 model
# reference from https://www.kite.com/python/answers/how-to-merge-two-pandas-series-into-a-dataframe-in-python
train_labels_m = train_feas.pop("math score")
train_labels_r = train_feas.pop("reading score")
train_labels_w = train_feas.pop("writing score")
train_labels = pd.concat([train_labels_m, train_labels_r, train_labels_w], axis=1)

test_labels_m = test_feas.pop("math score")
test_labels_r = test_feas.pop("reading score")
test_labels_w = test_feas.pop("writing score")
test_labels = pd.concat([test_labels_m, test_labels_r, test_labels_w], axis=1)

validation_labels_m = validation_feas.pop("math score")
validation_labels_r = validation_feas.pop("reading score")
validation_labels_w = validation_feas.pop("writing score")
validation_labels = pd.concat([validation_labels_m, validation_labels_r, validation_labels_w], axis=1)

normalizer = tf.keras.layers.Normalization(axis =-1)
normalizer.adapt(np.array(train_feas))

# tensor flow tutorial building training linear model.

## reference from https://www.tensorflow.org/tutorials/keras/regression
gender_female_based = np.array(train_feas['gender_female'])
gender_female_normalizer = layers.Normalization(input_shape=[1,], axis = None)
gender_female_normalizer.adapt(gender_female_based)

gender_female_model = tf.keras.Sequential([gender_female_normalizer, layers.Dense(units=1)])

gender_female_model.predict(gender_female_based[:10]) # predict build

gender_female_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

%time
history = gender_female_model.fit(train_feas['gender_female'], train_labels, epochs=12, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

test_results = {}
test_results['gender_female_model'] = gender_female_model.evaluate(test_feas['gender_female'], test_labels, verbose=0)



#Multiple inputs regression, reference https://www.tensorflow.org/tutorials/keras/regression

regression_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
regression_model.predict(train_feas[:10])
print(1)
regression_model.layers[1].kernel

regression_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')


%time
history = regression_model.fit(train_feas, train_labels, epochs=12, verbose=0, validation_split=0)
test_results['regression_model'] = regression_model.evaluate(test_feas, test_labels, verbose=0)

#/ Building Deep Training NN /#
## Following a tutorial on tensorflow.org.
def compile_func(para):
  network_model = keras.Sequential([para, layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)])
  network_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
  return network_model

## Single input Deep Neural
female_model_DN = compile_func(gender_female_normalizer)


## train the model and collect results references https://www.tensorflow.org/tutorials/keras/regression
# single input train set
%time
history = female_model_DN.fit(train_feas['gender_female'], train_labels, validation_split=0, verbose=0, epochs=12)
test_results['female_model_DN'] = female_model_DN.evaluate(test_feas['gender_female'], test_labels, verbose=0)

regression_model_DN = compile_func(normalizer)

%time
history = regression_model_DN.fit(train_feas, train_labels, validation_split=0, verbose=0, epochs=250)
test_results['regression_model_DN'] = regression_model_DN.evaluate(test_feas, test_labels, verbose=0)


## using Test set
test_regression_model_DN = compile_func(normalizer)
%time
history = test_regression_model_DN.fit(test_feas, test_labels, validation_split=0, verbose=0, epochs=250)
test_results['test_regression_model_DN'] = test_regression_model_DN.evaluate(validation_feas, validation_labels, verbose=0)



## using Validation set
validation_regression_model_DN = compile_func(normalizer)
%time
history = validation_regression_model_DN.fit(validation_feas, validation_labels, validation_split=0, verbose=0, epochs=250)
test_results['validation_regression_model_DN'] = validation_regression_model_DN.evaluate(validation_feas, validation_labels, verbose=0)

pd.DataFrame(test_results, index=['Margin of error [scores]']).T
