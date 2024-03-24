import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time



def train2(A,l):
    batch_size = 128
    num_epochs = 500
    expect_tho = 0.05
    Arm=torch.FloatTensor(A)
    emb = np.size(Arm, axis=1)
    def KL_devergence(p, q):
        """
        Calculate the KL-divergence of (p,q)
        :param p:
        :param q:
        :return:
        """
        q = torch.nn.functional.softmax(q, dim=0)
        q = torch.sum(q,
                      dim=0) / batch_size  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
        s1 = torch.sum(p * torch.log(p / q))
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        return s1 + s2

    model = AutoEncoder(emb=emb)
    if torch.cuda.is_available():
        model.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上

    Optimizer = optim.Adam(model.parameters(), lr=0.1)

    # 定义期望平均激活值和KL散度的权重
    tho_tensor = torch.FloatTensor([expect_tho for _ in range(32)])
    if torch.cuda.is_available():
        tho_tensor = tho_tensor.cuda()
    _beta = 0.1

    for epoch in range(num_epochs):
        time_epoch_start = time.time()
        encoder_out, decoder_out = model(Arm)
        attr_embedding = encoder_out.detach().numpy()
        if l==1:
            np.savetxt("./attr embedding_m.txt", attr_embedding)
        if l==0:
            np.savetxt("./attr embedding_r.txt", attr_embedding)
        if l==2:
            np.savetxt("./Sr_dis embedding.txt",attr_embedding)
        if l==3:
            np.savetxt("./Sm_dis embedding.txt",attr_embedding)
        loss = F.mse_loss(decoder_out, Arm)
        # 计算并增加KL散度到loss
        _kl = KL_devergence(tho_tensor, encoder_out)
        loss += _beta * _kl

        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))
    print("------------------------------------")
    # for epoch in range(num_epochs):
    #     time_epoch_start = time.time()
    #     encoder_out, decoder_out = model(A_r)
    #     attr_embedding = encoder_out.detach().numpy()
    #     np.savetxt("./attr embedding_r.txt", attr_embedding)
    #     loss = F.mse_loss(decoder_out, A_r)
    #     # 计算并增加KL散度到loss
    #     _kl = KL_devergence(tho_tensor, encoder_out)
    #     loss += _beta * _kl
    #     Optimizer.zero_grad()
    #     loss.backward()
    #     Optimizer.step()
    #     print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))


class AutoEncoder(nn.Module):
    def __init__(self,emb):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(emb, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,emb),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out

