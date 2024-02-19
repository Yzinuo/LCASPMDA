import numpy as np
import pandas as pd



def Fmatrix(M):
    A=M
    Z = np.loadtxt("./topo embedding1.txt")  # 节点拓扑表示矩阵 (1373+173)*128
    Z_r = Z[0:1373, :]  # 药物节点拓扑表示矩阵 1373*128
    Z_m = Z[1373:1546, :]  # 微生物节点拓扑表示矩阵 173*128
    A_rr = np.loadtxt("./attr embedding_r.txt")  # 药物节点属性表示矩阵 1373*32
    A_mm = np.loadtxt("./attr embedding_m.txt")  # 微生物节点属性表示矩阵 173*32
    Sr_che = np.loadtxt("Data/MDAD/drug_structure_sim.txt")  # S_r^Che drug structure similarity
    Sm_fun = np.loadtxt("Data/MDAD/microbe_function_sim.txt")  # S_m^f microbe function similarity
    Sr_dis=np.loadtxt("./Data/MDAD/Sr_dis_matrix.txt")
    Sm_dis=np.loadtxt("./Data/MDAD/Sm_dis_matrix.txt")

    Srr=np.loadtxt("./Data/MDAD/Srr.txt")
    Smm=np.loadtxt("./Data/MDAD/Smm.txt")
    Drug_feature = np.hstack((Z_r, np.hstack((A_rr, np.hstack((Sr_che, A))))))  # 1373*1706
    Drug_feature=np.hstack((np.hstack((Drug_feature,Sr_dis)),A))
    Drug_feature=np.hstack((np.hstack((Drug_feature,Srr)),A))
    Microbe_feature = np.hstack((Z_m, np.hstack((A_mm, np.hstack((A.T, Sm_fun))))))  # 173*1706
    Microbe_feature=np.hstack((np.hstack((Microbe_feature,A.T)),Sm_dis))
    Microbe_feature=np.hstack((np.hstack((Microbe_feature,A.T)),Smm))

    return  Drug_feature,Microbe_feature

