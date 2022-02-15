import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
# print(dataset.tail())

# print(dataset.isna().sum())
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

print(test_features['Horsepower'])

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')
print(test_labels)
# print(train_dataset.describe().transpose()[['mean', 'std']])


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

# first = np.array(train_features[:1])
#
# with np.printoptions(precision=2, suppress=True):
#     print('First example:', first)
#     print()
#     print('Normalized:', normalizer(first).numpy())

horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_test = np.array(test_features['Horsepower'])
horsepower_test_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_test_normalizer.adapt(horsepower_test)
horse_test = horsepower_test_normalizer(test_features['Horsepower'])

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

# print(horsepower_model.summary())
# horsepower_model.predict(horsepower[:10])

horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

# history = horsepower_model.fit(
#     train_features['Horsepower'],
#     train_labels,
#     epochs=100,
#     # Suppress logging.
#     verbose=0,
#     # Calculate validation results on 20% of the training data.
#     validation_split=0.2)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())

# print(history.history)

test_loss = horsepower_model.evaluate(test_features[['Horsepower']],  test_labels, verbose=2)
print('\nTest accuracy:', test_loss)


