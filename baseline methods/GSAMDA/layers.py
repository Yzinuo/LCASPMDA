import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F
from torch import nn


def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight=nn.Parameter(torch.Tensor(in_features,out_features))
        #self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #self.weight = torch.empty(in_features, out_features, requires_grad=True)
        #self.weight=glorot_init(in_features,out_features)
        #self.weight=torch.empty(in_features,out_features)
        #self.weight=glorot_init(in_features,out_features)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.num=num
        self.reset_parameters1(num)
        #self.reset_parameters(num)

    def reset_parameters2(self, num):
        self.weight = torch.nn.init.uniform_(self.weight, a=0.0, b=1.0)
        numm = str(num)
        np.savetxt("./Model parameters/weight" + numm + ".txt", self.weight.detach().numpy())
    def reset_parameters1(self,num):
        self.weight=torch.nn.init.kaiming_normal_(self.weight,mode='fan_in',nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        numm = str(num)
        np.savetxt("./Model parameters/weight"+numm+".txt",self.weight.detach().numpy())
    def reset_parameters(self,num):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        numm=str(num)
        np.savetxt("./Model parameters/weight"+numm+".txt",self.weight.detach().numpy())

    def forward(self, input, adj):
        #self.weight = torch.nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        #self.weight=torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        #self.reset_parameters1(self.num)
        #self.reset_parameters(self.num)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output+=self.bias.to(device)
        #
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu):
		super(GraphConvSparse, self).__init__()
		self.weight = glorot_init(input_dim, output_dim)
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


class GraphAttention(Module):
    def __init__(self,in_features,out_features,dropout,alpha,concat=True):
        super(GraphAttention, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.dropout=dropout
        self.alpha=alpha
        self.concat=concat

        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)

        self.leakyrelu=nn.LeakyReLU(self.alpha)
    def forward(self,inp,adj):
        h=torch.mm(inp,self.W)
        N=h.size()[0]
        a_input=torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)
        e=self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))

        zero_vec=-1e12*torch.ones_like(e)
        attention=torch.where(adj>0,e,zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime
    def __repr__(self):
        return self.__class__.__name__+' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

