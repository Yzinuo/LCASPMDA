from SAE_attention import *
from Feature_matrix_data import *
import random
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn import preprocessing
import numpy as np
from datadeal import *
from sklearn.metrics import precision_recall_curve

# 读入数据文件

A = np.loadtxt("./Data/MDAD/drug_microbe_matrix.txt")  # Adjacency matrx
Sr_che = np.loadtxt("Data/MDAD/drug_structure_sim.txt")  # S_r^Che drug structure similarity
Sm_f = np.loadtxt("Data/MDAD/microbe_function_sim.txt")  # S_m^f microbe function similarity
Sr_dis = np.loadtxt("Data/MDAD/Sr_dis_matrix.txt")
Sm_dis = np.loadtxt("Data/MDAD/Sm_dis_matrix.txt")
known = np.loadtxt("Data/MDAD/known.txt")  # 已知关联索引（序号从1开始）
unknown = np.loadtxt("Data/MDAD/unknown.txt")  # 未知关联索引（序号从1开始）


'''
A = np.loadtxt("./Data/aBiofilm/drug_microbe_matrix.txt")  # Adjacency matrx
Sr_che = np.loadtxt("Data/aBiofilm/drug_structure_sim.txt")  # S_r^Che drug structure similarity
Sm_f = np.loadtxt("Data/aBiofilm/microbe_function_sim.txt")  # S_m^f microbe function similarity
Sr_dis = np.loadtxt("./Data/aBiofilm/Sr_dis_matrix.txt")
Sm_dis = np.loadtxt("./Data/aBiofilm/Sm_dis_matrix.txt")
known = np.loadtxt("./Data/aBiofilm/known.txt")  # 已知关联索引（序号从1开始）
unknown = np.loadtxt("./Data/aBiofilm/unknown.txt")  # 未知关联索引（序号从1开始）
'''

# SM:基于药物微生物关联的药物相似性矩阵，
# 重启随机游走算法，估计两个节点之间的接近度
def RWR(SM):
    alpha = 0.1
    E = np.identity(len(SM))  # 单位矩阵
    M = np.zeros((len(SM), len(SM)))
    s = []
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

predict = []
TP, TN, FP, FN = 0, 0, 0, 0
scores = []
tlabels = []


# 5-fold cv  5折交叉验证
#
def kflod_5(num):
    k = []
    unk = []
    # 用作测试集各占20%
    lk = len(known)  # 已知关联数   2470
    luk = len(unknown)  # 未知关联数   235059
    for i in range(lk):
        k.append(i)
    for i in range(luk):
        unk.append(i)
    random.shuffle(k)  # 打乱顺序
    random.shuffle(unk)
    for cv in range(1, 6):
        interaction = np.array(list(A))  # 将邻接矩阵A转化为数组
        if cv < 5:
            B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/10的1的索引
            # print(B1.shape)   (494,2)
            B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/10的0的索引
            # print(B2.shape)    (47011,2)
            min_length = min(len(B1), len(B2))
            B1 = B1[:min_length, :]
            B2 = B2[:min_length, :]
            for i in range(lk // 10):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        else:
            B1 = known[k[(cv - 1) * (lk // 5):lk], :]
            B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
            min_length = min(len(B1), len(B2))
            B1 = B1[:min_length, :]
            B2 = B2[:min_length, :]
            for i in range(lk - (lk // 5) * 4):
                interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
        # interaction(1373,173)
        #Sr_m_HIP = HIP_Calculate(interaction)  # 药物的HIP相似性
        #Sm_r_HIP = HIP_Calculate(interaction.T)  # 微生物的HIP相似性
        Sm_r_GIP = GIP_Calculate(interaction)  # 微生物GIP相似性
        Sr_m_GIP = GIP_Calculate1(interaction)  # 药物GIP相似性
        #Sr_dis = Cosine_Sim(interaction)
        #Sm_dis = Cosine_Sim(interaction.T)
       # Sr_m = (Sr_dis + Sr_m_GIP ) / 2
       # Sm_r = (Sm_dis + Sm_r_GIP) / 2

        Srr = RWR(Sr_m_GIP)
        Smm = RWR(Sm_r_GIP)

        np.savetxt("./Data/MDAD/Srr.txt", Srr)  # 1373*1373
        np.savetxt("./Data/MDAD/Smm.txt", Smm)  # 173*173
        #np.savetxt("./Data/attr embedding/attr embedding_Sr_dis.txt", Sr_dis)
        #np.savetxt("./Data/attr embedding/attr embedding_Sm_dis.txt", Sm_dis)
        '''
        A_r1 = np.hstack((Srr, interaction))  # 1373*1546
        A_m1 = np.hstack((interaction.T, Smm))  # 173*1546
        A_r2 = np.hstack((Sr_che,interaction))  #1373 1546
        A_m2 = np.hstack((interaction.T,Sm_f))   #173 1546
        df = np.hstack((A_r1, A_r2)) #1546 1546
        mf = np.hstack((A_m1, A_m2)) #1546  1546
        '''
        train2(Sr_che ,2)
        train2(Sm_f ,3)
        train2(Sr_dis, 4)
        train2(Sm_dis, 5)

        #construct Feature matrix for drug-microbe node pair
        df, mf = Fmatri(A)  # 返回微生物药物矩阵# 1373*4894# 173*4894
        # 处理数据
        # 计算两者之间的预测分数 

        #df = np.loadtxt("./Data/attr embedding/attr embedding_A_r1.txt")
        #mf = np.loadtxt("./Data/attr embedding/attr embedding_A_m1.txt")
        #df = np.hstack((A_r1,A_r2))
        #mf = np.hstack((A_m1,A_m2))

        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)  # 1373*4702
        mf = min_max_scaler.fit_transform(mf)  # 173*4702

        score = torch.sigmoid(torch.FloatTensor(np.dot(df, mf.T)))  # 1373*173
        score = np.array(score)


        for i in range(len(B1)):  # know
            index1 = int(B1[i, 0] - 1)
            index2 = int(B1[i, 1] - 1)
            scores.append(score[index1, index2])
            tlabels.append(A[index1, index2])
        for i in range(len(B2)):  # unknow
            index1 = int(B2[i, 0] - 1)
            index2 = int(B2[i, 1] - 1)
            scores.append(score[index1, index2])
            tlabels.append(A[index1, index2])
        print("fold cv--{}".format(cv))
    #fpr, tpr, threshold = roc_curve(labels, score)
    num = str(num)
    #计算auc值
    fpr, tpr, threshold1 = roc_curve(tlabels, scores)
    precision, recall, threshold2 = precision_recall_curve(tlabels, scores)
    #num=str(num)
    np.savetxt("./Data/fpr_tpr/fpr"+num+".txt",fpr)
    np.savetxt("./Data/fpr_tpr/tpr"+num+".txt",tpr)
    auc_val=auc(fpr, tpr)
    #计算aupr值
    aupr_val = auc(recall, precision)  # 计算AUPR值
    np.savetxt("./Data/recall_pre/re" + num + ".txt", recall)
    np.savetxt("./Data/recall_pre/pr" + num + ".txt", precision)
    predict=[auc_val,aupr_val]
    print("auc:",auc_val)
    print("aupr:",aupr_val)

    return predict

auc_val = []
aupr_val = []
for i in range(10):
    predict = kflod_5(i)
    print("------------------------------")
    auc_val.append(predict[0])
    aupr_val.append(predict[1])
    np.savetxt("./Data/fpr_tpr/auc.txt",auc_val)
    np.savetxt("./Data/recall_pre/aupr.txt",aupr_val)
print("auc:",sum(auc_val)/10)
print("aupr:",sum(aupr_val)/10)