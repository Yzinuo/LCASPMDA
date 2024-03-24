from SAE import *
from Feature_matrix import *
from main1 import *
import random
import torch
from sklearn.metrics import roc_curve,auc,precision_recall_curve,precision_score,recall_score
from sklearn import preprocessing
from model1 import *



def accuracy(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    threshold_value = (outputs.max() + outputs.min()) /2.
    outputs = outputs.ge(threshold_value).type(torch.int32)
    labels = labels.type(torch.int32)
    corrects = (1 - (outputs ^ labels)).type(torch.int32)
    if labels.size() == 0:
        return np.nan
    return corrects.sum().item() / labels.size()[0]


def precision(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    threshold_value = (outputs.max() + outputs.min()) / 2.
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(threshold_value).type(torch.int32).detach().cpu().numpy()
    return precision_score(labels, outputs)


def recall(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    threshold_value = (outputs.max() + outputs.min()) / 2.
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(threshold_value).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs)

def f1(outputs, labels):
    return (precision(outputs, labels) + recall(outputs, labels)) / 2


def mcc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    threshold_value = (outputs.max() + outputs.min()) / 2.
    outputs = outputs.ge(threshold_value).type(torch.int32)
    labels = labels.type(torch.int32)
    true_pos = (outputs * labels).sum()
    true_neg = ((1 - outputs) * (1 - labels)).sum()
    false_pos = (outputs * (1 - labels)).sum()
    false_neg = ((1 - outputs) * labels).sum()
    numerator = true_pos * true_neg - false_pos * false_neg
    deno_2 = outputs.sum() * (1 - outputs).sum() * labels.sum() * (1 - labels).sum()
    if deno_2 == 0:
        return np.nan
    return (numerator / (deno_2.type(torch.float32).sqrt())).item()


A = np.loadtxt("./Data/aBiofilm/drug_microbe_matrix.txt")  # Adjacency matrx
Sr_che = np.loadtxt("Data/aBiofilm/drug_structure_sim.txt")  # S_r^Che drug structure similarity
Sm_f = np.loadtxt("Data/aBiofilm/microbe_function_sim.txt")  # S_m^f microbe function similarity
Sr_dis = np.loadtxt("./Data/aBiofilm/Sr_dis_matrix.txt")
Sm_dis = np.loadtxt("./Data/aBiofilm/Sm_dis_matrix.txt")
known = np.loadtxt("./Data/aBiofilm/known.txt")  # 已知关联索引（序号从1开始）
unknown = np.loadtxt("./Data/aBiofilm/unknown.txt")  # 未知关联索引（序号从1开始）
def RWR(SM):
    alpha = 0.1
    E = np.identity(len(SM))  # 单位矩阵
    M = np.zeros((len(SM), len(SM)))
    s=[]
    for i in range(len(M)):
        for j in range(len(M)):
            M[i][j] = SM[i][j] / (np.sum(SM[i, :]))
    for i in range(len(M)):
        e_i = E[i, :]
        p_i1 = np.copy(e_i)
        for j in range(10):
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s.append(p_i1)
    return s
def Net_construct(Sr_m,Sm_r):     #异构网络的构建
    N1=np.hstack((Sr_m,A))
    N2=np.hstack((A.T,Sm_r))
    Net=np.vstack((N1,N2))      #(1373+173)*(1373+173)
    return Net
TP,TN,FP,FN=0,0,0,0
scores=[]
tlabels=[]

#5-fold cv
def kflod_5(num):
    k = []
    unk = []
    lk = len(known)  # 已知关联数
    luk = len(unknown)  # 未知关联数
    for i in range(lk):
        k.append(i)
    for i in range(luk):
        unk.append(i)
    random.shuffle(k)  # 打乱顺序
    random.shuffle(unk)
    for cv in range(1, 6):
        interaction = np.array(list(A))
        if cv < 5:
            B1_indices = k[(cv - 1) * (lk // 5):(lk // 5) * cv]
            B1 = known[B1_indices, :]

            B2_indices = unk[(cv - 1) * len(B1_indices):(cv - 1) * len(B1_indices) + len(B1_indices)]
            B2 = unknown[B2_indices, :]
            for i in range(lk // 5):
               interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        else:
            B1_indices = k[(cv - 1) * (lk // 5):]
            B1 = known[B1_indices, :]

            B2_indices = unk[(cv - 1) * len(B1_indices):(cv - 1) * len(B1_indices) + len(B1_indices)]
            B2 = unknown[B2_indices, :]
            for i in range(lk - (lk // 5) * 4):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        # if cv < 5:
        #     B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/5的1的索引
        #     B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/5的0的索引
        #     for i in range(lk // 5):
        #         interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        # else:
        #     B1 = known[k[(cv - 1) * (lk // 5):lk], :]
        #     B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
        #     for i in range(lk - (lk // 5) * 4):
        #         interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        Sr_m_HIP = HIP_Calculate(interaction)  # 药物的HIP相似性
        Sm_r_HIP = HIP_Calculate(interaction.T)  # 微生物的HIP相似性
        Sm_r_GIP = GIP_Calculate(interaction)  # 微生物GIP相似性
        Sr_m_GIP = GIP_Calculate1(interaction)  # 药物GIP相似性
        Sr_m = (Sr_m_HIP + Sr_m_GIP) / 2
        Sm_r = (Sm_r_HIP + Sm_r_GIP) / 2
        # #Learning node topology representations
        Net = Net_construct(Sr_m, Sm_r)
        train33(Net, 0)
        print("------------------------------------------")
        # topo_emb=topo_feature(Net)          #提取出的节点topology特征矩阵 （1373+173)*128
        # #Learning node attribute representations
        Srr = RWR(Sr_m)
        Smm = RWR(Sm_r)
        np.savetxt("./Data/aBiofilm/Srr.txt", Srr)
        np.savetxt("./Data/aBiofilm/Smm.txt", Smm)
        A_r = np.hstack((np.hstack((np.hstack((interaction, Sr_che)), interaction)), Srr))  # 1373*(1546+109)
        A_m = np.hstack((np.hstack((np.hstack((interaction.T, Sm_f)), interaction.T)), Smm))  # 173*(1546+109)
        A_r = np.hstack((A_r, Sr_dis))
        A_m = np.hstack((A_m, Sm_dis))


        train2(A_r, 0)
        train2(A_m, 1)

        # construct Feature matrix for drug-microbe node pair
        df, mf = Fmatrix(A)

        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
        mf = min_max_scaler.fit_transform(mf)

        score = torch.sigmoid(torch.FloatTensor(np.dot(df, mf.T)))
        score = np.array(score)
        for i in range(len(B1)):
            index1 = int(B1[i, 0] - 1)
            index2 = int(B1[i, 1] - 1)
            scores.append(score[index1, index2])
            tlabels.append(A[index1, index2])
        for i in range(len(B2)):
            index1 = int(B2[i, 0] - 1)
            index2 = int(B2[i, 1] - 1)
            scores.append(score[index1, index2])
            tlabels.append(A[index1, index2])
        print("fold cv--{}".format(cv))
        fpr, tpr, threshold = roc_curve(tlabels, scores)
        num=str(cv)
        precision, recall, _ = precision_recall_curve(tlabels, scores)

        np.savetxt("./Data/fpr_tpr/fpr"+num+".txt",fpr)
        np.savetxt("./Data/fpr_tpr/tpr" + num + ".txt", tpr)
        np.savetxt("./Data/PR/precision" + num + ".txt", precision)
        np.savetxt("./Data/PR/recall" + num + ".txt", recall)

        auc_val = auc(fpr, tpr)
        PR = auc(recall, precision)

        predicts = torch.FloatTensor(scores)
        labels = torch.FloatTensor(tlabels)

        np.savetxt("./Data/PR/predicts" + num + ".txt", precision)
        np.savetxt("./Data/PR/labels" + num + ".txt", labels)

        acc = accuracy(predicts,labels)
        f1_s = f1(predicts,labels)
        mcc_s =mcc(predicts,labels)


        print(auc_val)
        print(PR)
        print(acc)
        print(f1_s)
        print(mcc_s)


auc_val=[]
aupr=[]
for i in range(10):
    a=kflod_5(i)
    print("------------------------------")
    auc_val.append(a)
    np.savetxt("./Data/fpr_tpr/auc.txt",auc_val)
    print(sum(auc_val)/10)

kflod_5(1)
