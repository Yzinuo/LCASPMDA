import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyg import GATv1Layer, GATv2Layer
from pyg import GAT2v1Layer, GAT2v2Layer
from pyg import GATPNAv1Layer, GATPNAv2Layer

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.001)
            # self.bias = self.bias * 1e42
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        # out = F.dropout(out,0.1)
        return self.act(out)



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels,
                 add_self_loops,
                 heads,
                 heads_aggr,
                 mode,
                 att_version,
                 ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.heads = heads
        self.heads_aggr = heads_aggr
        self.mode = mode
        self.att_version = att_version

        if self.att_version == 'v1':
            Lcat = GATv1Layer
        elif self.att_version == 'v2':
            Lcat = GATv2Layer
        self.lcat = Lcat(
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

    def forward(self, feats, sn_edge):
        lcat = self.lcat
        output = lcat(x=feats,
                      edge_index=sn_edge[0], size_target=None, edge_weight=sn_edge[1])
        return output


    # def get_embeds_sn(self, feats, nei_index, sn_edge):
    #
    #     h = F.elu(self.fc_list(feats))
    #     z_sn = self.sn(h, nei_index, sn_edge)
    #     return z_sn



class NN(nn.Module):
    def __init__(self, ninput,  noutput, nlayers, dropout=0.5):

        super(NN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput, noutput) for l in range(self.nlayers)])

    def forward(self, x):
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
        return x

class MDA_Decoder(nn.Module):
    def __init__(self, microbe_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(MDA_Decoder, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.drug_num = Drug_num
        self.microbe_num = microbe_num
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else Nodefeat_size, Nodefeat_size) for l in
            range(nlayers)])

        self.linear = torch.nn.Linear(Nodefeat_size, 1)

        self.drug_linear = torch.nn.Linear(Nodefeat_size, Nodefeat_size)
        self.microbe_linear = torch.nn.Linear(Nodefeat_size, Nodefeat_size)


    def forward(self, nodes_features, drug_index, microbe_index):

        microbe_features = nodes_features[microbe_index]
        drug_features = nodes_features[drug_index]

        microbe_features = self.microbe_linear(microbe_features)
        drug_features = self.drug_linear(drug_features)
        pair_nodes_features = drug_features*microbe_features
        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))

        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)

class MDA_Graph(nn.Module):

    def __init__(self,  nn_nlayers, hidden_dim,DTI_nn_nlayers,out_channels,
                             add_self_loops,
                             heads,
                             heads_aggr,
                             mode,
                             att_version, microbe_num, Drug_num, dropout, args_pre, feats_dim_list, train_type=1):
        super(MDA_Graph, self).__init__()
        # self.mp_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        self.sn_nn = NN(out_channels,  128, nn_nlayers, dropout)
        self.fc_list = nn.Linear(feats_dim_list, 4*out_channels, bias=True)
        self.MDA_Decoder = MDA_Decoder(microbe_num, Drug_num, 128, hidden_dim, DTI_nn_nlayers, dropout)
        self.LayerNorm = torch.nn.LayerNorm(128)

        self.encoder = Encoder(4*out_channels,out_channels,
                             add_self_loops,
                             heads,
                             heads_aggr,
                             mode,
                             att_version)
        # self.encoder.cuda()
        self.drug_num = Drug_num
        self.proein_num = microbe_num





    def forward(self, microbe_index, drug_index, feats,  sn_edge,):

        feats= F.elu(self.fc_list(feats))
        embs_sn = self.encoder(feats, sn_edge)
        embs_sn = self.sn_nn(embs_sn)
        embs_sn = self.LayerNorm(embs_sn)
        Nodes_features = embs_sn

        # Decoder
        output = self.MDA_Decoder(Nodes_features, drug_index, microbe_index)
        output = output.view(-1)


        return output, Nodes_features


