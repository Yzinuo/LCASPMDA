import torch
from torch import nn
from layers import GraphAttention
import torch.nn.functional as F
import numpy as np
from datadeal import *
import torch.optim as optim
import time
class GAT(nn.Module):
    def __init__(self,nfeat,nclass,dropout,alpha,l):
        super(GAT, self).__init__()
        self.gal=GraphAttention(nfeat,nclass,dropout,alpha)
        self.l=l
    def forward(self,x,adj):
        Z=self.gal(x,adj)
        a=Z.detach().numpy()
        if self.l==0:
            np.savetxt("./topo embedding1.txt",a)
        if self.l==1:
            np.savetxt("./attr embedding1.txt",a)
        if self.l == 2:
            np.savetxt("./Srm_dis embedding.txt", a)
        # if self.l == 3:
        #     np.savetxt("./Sm_dis embedding.txt", a)
        ZZ=torch.sigmoid(torch.matmul(Z,Z.T))
        return ZZ


def Net_construct(Sr_m,Sm_r):     #异构网络的构建
    N1=np.hstack((Sr_m,A))
    N2=np.hstack((A.T,Sm_r))
    Net=np.vstack((N1,N2))      #(1373+173)*(1373+173)
    return Net

def train33(Net,l):
    adj=torch.FloatTensor(Net)
    x=torch.FloatTensor(Net)
    idx_train = range(1300)
    idx_test = range(1300, 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    model=GAT(x.shape[1],128,0.4,0.2,l)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=5e-4)
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(x, adj)
        loss_train = F.mse_loss(output[idx_train,:], adj[idx_train,:])
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.5f}'.format(loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_train

    def test():
        model.eval()
        output = model(x, adj)
        loss_test = F.mse_loss(output[idx_test,:], adj[idx_test,:])
        print("Test set results:",
              "loss= {:.5f}".format(loss_test.item()))

    t_total = time.time()
    for epoch in range(1):
        loss = train(epoch)
        # if loss < 0.1:
        #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.5f}s".format(time.time() - t_total))

    test()







