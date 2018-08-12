# I will try to verify the universal approximation theorem on an arbitrary function 

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim

def f_x(x):
    return max(0,x) # random function to learn


# Building dataset
def build_dataset():
    # Given f(x), is_f_x defines whether the function is satisfied
    data = []
    for i in range(-99,0):
        data.append((i,f_x(i), 1)) # True
    for j in range(0, 101):
        data.append((j,f_x(j)+j*j, 0)) # Not true
    column_names = ["x","f_x", "is_f_x"]
    df = pd.DataFrame(data, columns=column_names)
    return df

df = build_dataset()
print ("Dataset is built!")

    
labels = df.is_f_x.values
features = df.drop(columns=['is_f_x']).values

print ("shape of features:", features.shape)
print ("shape of labels: ", labels.shape)


# Building nn
net = nn.Sequential(nn.Linear(features.shape[1],100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(),nn.Linear(100, 2))

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=34)

# parameters
optimizer = optim.Adam(net.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
epochs=300


def train():
    net.train()
    losses = []
    for epoch in range(1,200):
        x_train = Variable(torch.from_numpy(features_train)).float()
        y_train = Variable(torch.from_numpy(labels_train)).long()
        y_pred = net(x_train)
        loss = criterion(y_pred, y_train)
        print ("epoch #", epoch)
        print (loss.item())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses


print ("training start....")
losses = train()
plt.plot(range(1, 200), losses)
plt.xlabel("epoch")
plt.ylabel("loss train")
plt.show()

print ("testing start ... ")
x_test = Variable(torch.from_numpy(features_test)).float()
x_train = Variable(torch.from_numpy(features_train)).float()


def test():
    pred = net(x_test)
    pred = torch.max(pred, 1)[1]
    print ("Accuracy on test set: ", accuracy_score(labels_test, pred.data.numpy()))

    p_train = net(x_train)
    p_train = torch.max(p_train, 1)[1]
    print ("Accuracy on train set: ", accuracy_score(labels_train, p_train.data.numpy()))
           
test()
