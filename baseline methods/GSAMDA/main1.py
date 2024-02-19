import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import torch.autograd.variable as Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from Feature_matrix import Fmatrix
from sklearn.model_selection import train_test_split
from datadeal import *

from layers import GraphConvolution

def train4(F_train,label_train,F_test,tlabel_test):
    nepoch = 2
    predict = []
    x_train, x_test, y_train, y_test = F_train, F_test, label_train, tlabel_test

    class Convnet(nn.Module):
        def __init__(self, nclass=2):
            super(Convnet, self).__init__()
            self.layer1 = nn.Sequential(
                GraphConvolution(1706,256,num=3),
                #nn.BatchNorm2d(3),  # 正则化
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                GraphConvolution(256,64,num=4),
                #nn.BatchNorm2d(6),
                nn.ReLU()
                #nn.MaxPool2d(kernel_size=1, stride=1)
            )
            self.fc = nn.Linear(256, nclass)

        def forward(self, x,adj):
            out = self.layer1(x,adj)
            out = self.layer2(out,adj)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Convnet(2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train
    total_step = len(x_train)
    for epoch in range(nepoch):
        for i in range(len(x_train)):
            inputs, label=x_train[i],y_train[i]
            #inputs = inputs.to(torch.float32)
            # inputs=torch.tensor(item for item in inputs)
            # labels=torch.tensor(item for item in labels)
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            adj = sp.coo_matrix(inputs)
            features = sp.csr_matrix(inputs)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            features = torch.FloatTensor(np.array(features.todense()))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            outputs = model(features,adj)
            labels = labels.long()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}'
                      .format(epoch + 1, nepoch, i + 1, total_step, loss.item()))
    # Test the model
    model.eval()
    score = []
    tlabel = []
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(len(x_test)):
            inputs,labels = x_test[i],y_test[i]
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            labels = labels.long()
            outputs = model(inputs)
            data = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(data, 1)
            total += labels.size(0)
            tmp = data.numpy()
            tmp1 = labels.numpy()
            for i in range(len(tmp)):
                score.append(tmp[i][1])
                tlabel.append(tmp1[i])
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    return score, tlabel

# A=np.loadtxt("./Data/MDAD/drug_microbe_matrix.txt")
# x,y=Fmatrix(A)
# train4(x,y,x,y)