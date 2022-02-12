import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.array(np.load('data.npy'))
data_results = np.array(pd.read_csv('train.csv')['Survived'])
train_data, test_data, train_results, test_results = train_test_split(data, data_results, shuffle = False)

opt = Adam(learning_rate = 0.01)
model = Sequential()
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss = 'BinaryCrossentropy', optimizer = opt, metrics = ['acc'])

fit_results = model.fit(train_data, train_results, epochs = 150,validation_split = 0.2)
model.save_weights('models/tensorflow_ANN_model')
plt.plot(range(150), fit_results.history['acc'], label = 'acc')
plt.plot(range(150), fit_results.history['val_acc'], label = 'val_acc')
plt.plot(range(150), fit_results.history['val_loss'], label = 'val_loss')
plt.plot(range(150), fit_results.history['loss'], label = 'loss')
plt.legend()
plt.show()

model.load_weights('models/tensorflow_ANN_model')
results = pd.DataFrame(columns = ('predicted', 'true'))
results['predicted'] = model.predict(test_data).flatten()
results['predicted'] = round(results['predicted'])
results['true'] = pd.DataFrame(test_results)
correct = (results['predicted'] == results['true']).sum()

print(f'accuracy:{100 * correct/len(test_results)}')

results.to_csv('results/tensorflow_model_results')
