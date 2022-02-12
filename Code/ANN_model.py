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


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.silu3 = nn.SiLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.silu3(out)
        out = self.fc4(out)
        return out

batch_size = 100
num_iters = 400000
epochs = num_iters / (len(features_train) / batch_size)
epochs = int(epochs)

input_dim = 7
hidden_dim = 4
output_dim = 1

model = ANNModel(input_dim, hidden_dim, output_dim)

error = nn.MSELoss()
learning_rate = 0.001
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
    if (epoch % 2000 == 0):
        pred = round(pd.DataFrame(model(featuresTest).detach().numpy()))
        correct = (pred == answerTest).sum()
        accuracy = int(100 * correct / total)
        acc_list.append(accuracy)
        loss_list.append(loss.data)
        iter_list.append(epoch)
        print(f'epoch: {epoch}, loss: {loss.data}, accuracy: {accuracy}%')

torch.save(model, 'models/ANN_model_torch')

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

model = torch.load('models/ANN_model_torch')
results = pd.DataFrame(columns=('predictions', 'true'))
results['true'] = answerTest.flatten()
results['predictions'] = model(featuresTest).detach().numpy().flatten()
correct = (round(results['predictions']) == results['true']).sum()
print(f'accuracy:{int(100 * (correct) / len(answerTest))}%')
results.head()
results.to_csv('results/results_ANN_model_torch')
