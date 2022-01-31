import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings(action='ignore')

data_features = np.array(pd.DataFrame(np.load('data.npy')), dtype = np.float32).reshape(-1, 7)
data_answer = np.array(pd.read_csv('train.csv').loc[:, pd.read_csv('train.csv').columns == 'Survived'], dtype = np.float32).reshape(-1,1)

features_train, features_test, answer_train, answer_test = train_test_split(data_features,
                                                                           data_answer,
                                                                           test_size = 0.2,
                                                                           random_state = 42)
featuresTrain = Variable(torch.from_numpy(features_train))
answerTrain = Variable(torch.from_numpy(answer_train))
featuresTest = Variable(torch.from_numpy(features_test))
answerTest = answer_test


class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

batch_size = 100
num_iters = 10000
epochs = num_iters / (len(features_train) / batch_size)
epochs = int(epochs)

input_dim = 7
output_dim = 1

model = RegressionModel(input_dim, output_dim)

error = nn.MSELoss()
learning_rate = 0.001
optim = torch.optim.RMSprop(model.parameters(), lr = learning_rate)

count = 0
loss_list = []
acc_list = []
iter_list = []
total = len(featuresTest)
for epoch in range(epochs):
    optim.zero_grad()
    results = model(featuresTrain)
    loss = error(results, answerTrain)
    loss.backward()
    optim.step()
    if (epoch % 100 == 0):
        pred = round(pd.DataFrame(model(featuresTest).detach().numpy()))
        correct = (pred == answerTest).sum()
        accuracy = int(100 * correct / total)
        acc_list.append(accuracy)
        loss_list.append(loss.data)
        iter_list.append(epoch)
        print(f'epoch: {epoch}, loss: {loss.data}, accuracy: {accuracy}%')

torch.save(model, 'models/regression_model_torch')

plt.plot(iter_list, loss_list)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title('loss')
plt.show()
plt.plot(iter_list, acc_list, color='red')
plt.xlabel("epochs")
plt.ylabel("acc")
plt.title('acc')
plt.show()

model = torch.load('models/regression_model_torch')
results = pd.DataFrame(columns = ('predictions', 'true'))
results['true'] = answerTest.flatten()
results['predictions'] = model(featuresTest).detach().numpy().flatten()
correct = (round(results['predictions']) == results['true']).sum()
print(f'accuracy:{int(100 * (correct)/len(answerTest))}%')
results.head()
results.to_csv('results/results_regression_model_torch')