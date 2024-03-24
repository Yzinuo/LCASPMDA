import torch
from torch import nn
from layers import GraphAttention
import torch.nn.functional as F
import numpy as np
from datadeal import *
import torch.optim as optim
import time
from pyg import GATv1Layer, GATv2Layer
from pyg import GAT2v1Layer, GAT2v2Layer
from pyg import GATPNAv1Layer, GATPNAv2Layer
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


def train33(Net,l):
    adj=torch.FloatTensor(Net)
    x=torch.FloatTensor(Net)
    idx_train = range(1300)
    idx_test = range(1300, 1500)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    model=LCAT(x.shape[1],128,True, 1, 'mean', 'lcat', 'v1')
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
    for epoch in range(200):
        loss = train(epoch)
        # if loss < 0.1:
        #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.5f}s".format(time.time() - t_total))

    test()

class LCAT(nn.Module):
    def __init__(self,in_channels,out_channels,
                             add_self_loops,
                             heads,
                             heads_aggr,
                             mode,
                             att_version,
                             ):
        super(LCAT, self).__init__()
        self.in_channels  =in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.heads =  heads
        self.heads_aggr = heads_aggr
        self.mode = mode
        self.att_version = att_version


    def forward(self, feats, sn_edge):

        if self.att_version == 'v1':
            LCAT = GATv1Layer
        elif self.att_version == 'v2':
            LCAT = GATv2Layer
        lcat = LCAT(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            negative_slope=0.2,
            add_self_loops=self.add_self_loops,
            heads=self.heads,
            bias=True,
            mode=self.mode,
            share_weights_score=False,
            shere_weights_value=False,
            aggr='mean',
        )

        output = lcat(x=feats,
                      edge_index=sn_edge[0],size_target = None,edge_weight = sn_edge[1] )
        return output




