import torch as th
from torch import nn
from torch import optim
import numpy as np
import time
#from ResidualBlock import *
import torch.nn.functional as F
#from External_Attention import *
from SelfAttention import *
#逐层预训练
def trainAE(encoderList, trainLayer, Arm, epoch,emb,l,useCuda = False):
    if useCuda:
        for i in range(len(encoderList)):
            encoderList[i].cuda()
    optimizer = optim.Adam(encoderList[trainLayer].parameters(), lr=0.001)
    ceriation = nn.MSELoss()
  # trainLoader, testLoader = loadMNIST(batchSize=batchSize)

    for i in range(epoch):

        #sum_loss = 0
        if trainLayer != 0: # 单独处理第0层，因为第一个编码器之前没有前驱的编码器了
            for i in range(trainLayer): # 冻结要训练前面的所有参数
                for param in encoderList[i].parameters():
                    param.requires_grad = False

       # x, target = Variable(x), Variable(Arm)
       # out = Arm.view(-1,1373*4465)    #转变成一阶张量
        out = Arm.view(-1,emb)
        # 产生需要训练层的输入数据,对前（trainLayer-1)冻结了的层进行前向计算
        if trainLayer != 0:
            for j in range(trainLayer):
                out = encoderList[j](out, rep=True)
        # 训练指定的自编码器
        pred = encoderList[trainLayer](out, rep=False)
        # np.savetxt("./interaction.txt", attr_embedding)

        loss = ceriation(pred, out)
        #sum_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('==>>> train layer:{}, epoch: {}, train loss: {:.6f}'.format(trainLayer+1, i,  loss))

def trainClassifier(model, Arm, epoch, emb,l,useCuda = False):
    #batchsize = 128
    if useCuda:
        model = model.cuda()
    # 解锁参数
    for param in model.parameters():
        param.requires_grad = True

    #optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #ceriation = nn.L1Loss()
    ceriation = nn.MSELoss()

    #trainLoader, testLoader = loadMNIST(batchSize=batchSize)

    for i in range(epoch):
        # trainning
        #sum_loss = 0
        time_epoch_start = time.time()
        x = Arm.view(-1,emb)
        out = model(x)
        attr_embedding = out.detach().numpy()
        #np.savetxt("./interaction.txt", attr_embedding)
        '''
        if l == 1:
            np.savetxt("./attr embedding_m.txt", attr_embedding)
        if l == 0:
            np.savetxt("./attr embedding_r.txt", attr_embedding)
        '''
        if l == 2:
            np.savetxt("./Data/attr embedding/attr embedding_Sr_che", attr_embedding)
        if l == 3:
            np.savetxt("./Data/attr embedding/attr embedding_Sm_f", attr_embedding)

        if l == 4:
            np.savetxt("./Data/attr embedding/attr embedding_Sr_dis.txt", attr_embedding)
        if l == 5:
            np.savetxt("./Data/attr embedding/attr embedding_Sm_dis.txt", attr_embedding)

        loss = ceriation(out, Arm)
        #sum_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('==>>> epoch: {}, train loss: {:.6f},Time:{:.2f}'.format( i, loss,time.time() - time_epoch_start))

def train2(A,l):
    #batchSize = 128
    AEepoch = 100
    epoch = 100
    Arm = th.FloatTensor(A)  #1546*4638
    # arm.shape:(1373,4465)
    emb = np.size(Arm, axis=1)  # emb:4638
    #是否堆叠的越多，训练的效果就越好？
    encoder1 = AutoEncoder(emb, 128)
    encoder2 = AutoEncoder(128, 64)
    encoder3 = AutoEncoder(64, 32)
    #print(encoder3)
   # encoder4 = AutoEncoder(64,32)   ##

    '''
    每个自编码器都只是优化一层隐藏层，每个隐藏层的参数都只是局部最优
    优化完每个自编码器之后，把优化后的网络参数作为神经网络的初始值，
    之后进行整个网络的训练，直到网络收敛
    '''
    encoderList = [encoder1, encoder2,encoder3]
    for i in range(3):
        trainAE(encoderList, i, Arm, AEepoch,emb,l, useCuda=False)  #逐层预训练
    #trainAE(encoderList, 1, Arm, AEepoch,emb, useCuda=False)
    #trainAE(encoderList, 2, batchSize, AEepoch, useCuda=True)

    model = SAE(encoderList,emb)
    trainClassifier(model, Arm, epoch, emb,l,useCuda=False)     #微调模型


class AutoEncoder(nn.Module):
    '''
    用于堆叠式自动编码器的全连接线性层。
    当每一层都启用训练时，该模块可以自动进行训练
    最简单的自动编码器
    '''
    def __init__(self, inputDim, hiddenDim):
        super().__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.encoder = nn.Linear(inputDim, hiddenDim, bias=True)
        self.decoder = nn.Linear(hiddenDim, inputDim, bias=True)
        self.sa = SelfAttention(num_attention_heads=8, input_size=hiddenDim, hidden_size=hiddenDim)
        self.act = nn.ReLU()  ##

    def forward(self, x, rep=False):
        hidden = self.encoder(x)
        hidden = self.act(hidden)
        #hidden = self.sa(hidden)
        if rep == False:  #当逐层训练时，需要输出decoder的计算结果，作为一个输出层
            out = self.decoder(hidden)
            #out = self.act(out)
            return out
        else:
            return hidden

class SAE(nn.Module):
    #使用encoderList构建整个网络
    def __init__(self, encoderList,emb):

        super().__init__()

        self.encoderList = encoderList
        self.en1 = encoderList[0]
        self.en2 = encoderList[1]
        self.en3 = encoderList[2]
        #self.en4 = encoderList[3]

        self.fc = nn.Sequential(
            nn.Linear(32, emb, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(emb)
            #nn.Dropout(0.5)
        )

        #self.ea = ExternalAttention(d_model=64, S=16)
        self.sa2 = SelfAttention(num_attention_heads=4, input_size=32, hidden_size=32)

    def forward(self, x):
        out = x
        out = self.en1(out, rep=True)
        out = self.en2(out, rep=True)
        out = self.en3(out, rep=True)
        #out = self.en4(out, rep=True)
        #out = self.rblock1(out)
        #out = self.sa1(out)
        out = self.sa2(out)
        out = self.fc(out)
        #out = self.ea(49,50,out)
        #out = F.log_softmax(out)   #做归一化，softmax分类器

        return out