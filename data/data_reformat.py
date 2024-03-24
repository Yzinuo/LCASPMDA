import os.path
import random
from sklearn.model_selection import KFold
import numpy as np
import scipy.sparse as sp
from utils import *
import pandas as pd,math

def sc_data(adj,D,M,parent_dir,ds,ms,parent_adj):
    """
    train data set adj
    """
    real_adj = np.loadtxt(parent_adj)

    for i in real_adj:
        i[1] -= 1
        i[0] -= 1

    dm = np.array(adj)
    adj_m = np.array(real_adj)

    path_dm = os.path.join(parent_dir, "dm.txt")
    np.savetxt(path_dm, dm, fmt="%d")
    dm = sp.csr_matrix((dm[:, 2], (dm[:, 0], dm[:, 1])), shape=(D + M, D + M)).toarray()

    A = sp.csr_matrix((adj_m[:, 2], (adj_m[:, 0], adj_m[:, 1])), shape=(D , M)).toarray()
    adj_similiarty = np.vstack((np.hstack((ds, A)), np.hstack((A.T,ms))))
    adj_dm = dm + dm.transpose()
    path_sim = os.path.join(parent_dir, "adj_similiarty.npy")
    path_dm = os.path.join(parent_dir, "adj_dm.npy")
    np.save(path_sim, adj_similiarty)
    np.save(path_dm, adj_dm)

    return dm

"""
    construct meta path
    dmd、mdm、dmdmd、mdmdm
"""
def mp_data(parent_dir,dm,D,M):
    """
    construct meta-path dmd, mdm
    dmd: dm * md
    mdm: md * dm
    """
    md = dm.transpose()
    dmd = np.matmul(dm, md)
    dmd = sp.coo_matrix(dmd)
    # dmd.data = np.ones(dmd.data.shape[0])
    dmd1 = dmd.toarray()[:D,:D]
    dmd1 = sp.coo_matrix(dmd1)
    path_dmd = os.path.join(parent_dir, "dmd.npz")
    sp.save_npz(path_dmd, dmd1)

    mdm = np.matmul(md, dm)
    mdm = sp.coo_matrix(mdm)
    # mdm.data = np.ones(mdm.data.shape[0])
    mdm1 = mdm.toarray()[D:,D:]
    mdm1 = sp.coo_matrix(mdm1)
    path_mdm = os.path.join(parent_dir, "mdm.npz")
    sp.save_npz(path_mdm, mdm1)

    return dmd, mdm


def mp2_data(parent_dir,dm,dmd, mdm, D,M):
    """
        construct meta-path dmdmd, mdmdm
        dmdmd: dmd * dmd
        mdmdm: mdm * mdm
    """
    dm = dm[:D,D:]
    md = dm.transpose()
    dmd = dmd.toarray()
    mdm = mdm.toarray()
    d_m_d = dmd[:D,:D]
    m_d_m = mdm[D:,D:]
    dmdm = np.matmul(d_m_d , dm)
    dmdmd = np.matmul(dmdm , md)
    dmd_index = np.where(d_m_d > 0)
    dmdmd[dmd_index] = 0
    dmdmd = sp.coo_matrix(dmdmd)

    mdmd = np.matmul(m_d_m , md)
    mdmdm = np.matmul(mdmd , dm)
    mdm_index = np.where(m_d_m > 0)
    mdmdm[mdm_index] = 0
    mdmdm = sp.coo_matrix(mdmdm)

    path_dmdmd = os.path.join(parent_dir, "dmdmd.npz")
    sp.save_npz(path_dmdmd, dmdmd)
    path_mdmdm = os.path.join(parent_dir, "mdmdm.npz")
    sp.save_npz(path_mdmdm, mdmdm)

    return dmdmd, mdmdm





"""
    construct pos sample for contrastive learning
    each node has 10 pos sample
"""
def mp_pos(dmd, mdm,parent_dir,D,M,pos_num):

    dmd = dmd.A.astype("float32")
    dia_d = sp.dia_matrix((np.ones(D+M), 0), shape=(D+M, D+M)).toarray()



    dd = np.ones((D+M, D+M)) - dia_d
    dmd = dmd * dd
    pos_d = np.zeros((D, D))
    k = 0
    for i in range(D):
        """
        for each node, first select itself as a pos pair
        then select the top (10-1) meta-path based neighbor node as pos pair 
        """
        pos_d[i, i] = 1
        one = dmd[i].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-dmd[i, one])
            sele = one[oo[:pos_num - 1]]
            pos_d[i, sele] = 1
            k += 1
        else:
            pos_d[i, one] = 1
    pos_d = sp.coo_matrix(pos_d)
    path_pos_d = os.path.join(parent_dir, "pos_d.npz")
    sp.save_npz(path_pos_d, pos_d)


    mdm = mdm.A.astype("float32")
    # microbe
    dia_m = sp.dia_matrix((np.ones(M+D), 0), shape=(M+D, M+D)).toarray()
    mm = np.ones((M+D, M+D)) - dia_m
    mdm = mdm * mm
    pos_m = np.zeros((M, M))
    k = 0

    for j in range(M):
        pos_m[j, j] = 1
        i = j + D
        one = mdm[i].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-mdm[i, one])
            sele = one[oo[:pos_num - 1]]
            pos_m[j, sele-D] = 1
            k += 1
        else:
            pos_m[j, one-D] = 1
    pos_m = sp.coo_matrix(pos_m)
    path_pos_m = os.path.join(parent_dir, "pos_m.npz")
    sp.save_npz(path_pos_m, pos_m)

"""
5-fold 
"""
def cross_5_folds(adj, nodefeatures,dataset,drugfeat,microbefeat,seed=1):
    folds = np.ones(adj.shape[0])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = kf.split(folds)
    a = []
    b = [-1, -1]
    for i, data in enumerate(kf):
        a.append(data[1])

    for i in range(5):
        for j in range(len(a[i])):
            folds[a[i][j]] = i

    adj = adj.astype(np.int32).tolist()
    num = 0
    drug_id_list = list(range(len(drugfeat)))
    microbe_id_list = list(range(len(microbefeat)))
    pos_num = len(adj)

    """
    generate negative samples 1：1
    """
    temp_adj = []
    folds1 = []
    while num < pos_num:
        drug_id = random.choice(drug_id_list)
        microbe_id = random.choice(microbe_id_list)
        neg_pos1 = [drug_id, microbe_id, 1]
        neg_pos = [drug_id, microbe_id, 0]
        if (neg_pos1 in adj) or (neg_pos in temp_adj):
            continue
        temp_adj.append(adj[num])
        folds1.append(folds[num % pos_num])
        temp_adj.append(neg_pos)
        folds1.append(folds[num % pos_num])
        num += 1

    folds1 = np.array(folds1)
    temp_adj = np.array(temp_adj)
    path_data = os.path.join(dataset,"data_"+dataset+".npz")
    np.savez(path_data, drugfeat=drugfeat, microbefeat=microbefeat, adj=temp_adj, folds=folds1, nodefeat=nodefeatures)


def feat_combine(dm,adj,D,M,drugfeat,microbefeat,parent_dir):
    real_adj = adj
    for i in real_adj:
        i[1] -= D
    adj_m = np.array(real_adj)
    A = sp.csr_matrix((adj_m[:, 2], (adj_m[:, 0], adj_m[:, 1])), shape=(D, M)).toarray()
    nodefeatures = np.vstack((np.hstack((drugfeat, A)),np.hstack((A.T, microbefeat))))
    path_nodefeatures = os.path.join(parent_dir,"nodefeatures.npy")
    np.save(path_nodefeatures, nodefeatures)
    return nodefeatures

def GIP_Calculate(M):     #计算高斯核相似性
    l=np.size(M,axis=1)
    sm=[]
    m=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[:,i]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[:,i]-M[:,j]))**2))
    return m
def GIP_Calculate1(M):     #计算高斯核相似性
    l=np.size(M,axis=0)
    sm=[]
    m=np.zeros((l,l))
    km=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[i,:]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[i,:]-M[j,:]))**2))
    for i in range(l):
        for j in range(l):
            km[i,j]=1/(1+np.exp(-15*m[i,j]+math.log(9999)))
    return km

def processdata_encoder(dataset, train_positive_inter_pos,val_positive_inter_pos, pos_num ):
    # seed = 1
    # random.seed(seed)
    cv_i = 0
    np.random.seed(1)
    for pos_adj, pos_adj_val in zip(train_positive_inter_pos, val_positive_inter_pos):

        pos_adj = np.array(pos_adj)
        adj = np.ones([len(pos_adj),3])
        adj[:,0] = np.copy(pos_adj[:,0])
        adj[:,1] = np.copy(pos_adj[:,1])
        pos_num = pos_num  # pos sample num for contrastive learning
        root_dir = os.path.join("./data", dataset)
        parent_adj = os.path.join(root_dir, "adj.txt" )
        parent_dir = os.path.join(root_dir,'encoder_' + str(cv_i) )
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        path_ds = os.path.join(root_dir, "drugsimilarity.txt") ###
        ds = np.loadtxt(path_ds)
        path_ms = os.path.join(root_dir, "microbesimilarity.txt")###
        ms = np.loadtxt(path_ms)

        D = len(ds)
        M = len(ms)

        adj_temp = np.array(adj)
        adj_temp = sp.csr_matrix((adj_temp[:, 2], (adj_temp[:, 0], adj_temp[:, 1])), shape=(D, M)).toarray()
        interaction = np.array(list(adj_temp))
        Sm_r_GIP = GIP_Calculate(interaction)  # 微生物GIP相似性
        Sr_m_GIP = GIP_Calculate1(interaction)  # 药物GIP相似性
        path_drugfeat_GIP = os.path.join(parent_dir, "drug_GIP.txt")
        path_microbefeat_GIP = os.path.join(parent_dir, "microbe_GIP.txt")
        np.savetxt(path_drugfeat_GIP, Sm_r_GIP)
        np.savetxt(path_microbefeat_GIP, Sr_m_GIP)

        # drug_ids = set()
        # microbe_ids = set()

        # # 遍历 pos_adj_val 来填充集合
        # for drug_id, microbe_id in pos_adj_val:
        #     drug_ids.add(drug_id)
        #     microbe_ids.add(microbe_id)
        #
        # # 假设 ds 和 ms 是 numpy 数组
        # ds = np.array(ds)  # 药物互相关联矩阵
        # ms = np.array(ms)  # 微生物互相关联矩阵
        #
        # # 将测试集中的所有药物在 ds 中的数值清零
        # for drug_id in drug_ids:
        #     ds[drug_id, :] = 0
        #     ds[:, drug_id] = 0
        #
        # # 将测试集中的所有微生物在 ms 中的数值清零
        # for microbe_id in microbe_ids:
        #     ms[microbe_id, :] = 0
        #     ms[:, microbe_id] = 0


        for i in adj:
            i[1] += D



        adj = adj.astype("int")
        path_drugfeat = os.path.join(root_dir, "drugfeatures.txt")
        path_microbefeat = os.path.join(root_dir, "microbefeatures.txt")
        drugfeat = np.loadtxt(path_drugfeat)
        microbefeat = np.loadtxt(path_microbefeat)
        np.savetxt(os.path.join(parent_dir, "microbefeatures.txt"), microbefeat)
        np.savetxt(os.path.join(parent_dir, "adj.txt"), adj)
        np.savetxt(os.path.join(parent_dir, "drugsimilarity.txt"), ds)
        np.savetxt(os.path.join(parent_dir, "microbesimilarity.txt"), ms)
        np.savetxt(os.path.join(parent_dir, "drugfeatures.txt"), drugfeat)


        """
        construct train data for each 5-CV train set
        """
        dm = sc_data(adj,D,M,parent_dir,ds,ms,parent_adj)
        # nodefeatures = feat_combine(dm,adj,D,M,drugfeat,microbefeat,parent_dir)
        # cross_5_folds(adj, nodefeatures,dataset,drugfeat,microbefeat,seed)
        cv_i += 1




