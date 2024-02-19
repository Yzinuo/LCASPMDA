import numpy as np
import pandas as pd



def Fmatri(M):
    A=M   #1373*173
    #读取txt文件

    A_rr = np.loadtxt("./Data/MDAD/Srr.txt")  # 药物节点属性表示矩阵 1373*256
    A_mm = np.loadtxt("./Data/MDAD/Smm.txt")  # 微生物节点属性表示矩阵 173*256
    Sr_che = np.loadtxt("./Data/attr embedding/attr embedding_Sr_che.txt")  # S_r^Che drug structure similarity
    Sm_fun = np.loadtxt("./Data/attr embedding/attr embedding_Sm_f.txt")  # S_m^f microbe function similarity
    Sr_m=np.loadtxt("./Data/attr embedding/attr embedding_Sr_dis.txt")
    Sm_r=np.loadtxt("./Data/attr embedding/attr embedding_Sm_dis.txt")
    
    #Srr=np.loadtxt("../Data/MDAD/Srr.txt")
    #Smm=np.loadtxt("../Data/MDAD/Smm.txt")
   
    
    #将两个数组按水平方向组合起来
    Drug_feature =  np.hstack((Sr_che, A))   # 1373*1802
    Drug_feature = np.hstack((np.hstack((Drug_feature,Sr_m)),A))   #1373*3348
    Drug_feature = np.hstack((np.hstack((Drug_feature,A_rr)),A))
    Microbe_feature = np.hstack((A.T, Sm_fun))  # 173*1802
    Microbe_feature = np.hstack((np.hstack((Microbe_feature,A.T)),Sm_r))   #173*3348
    Microbe_feature = np.hstack((np.hstack((Microbe_feature,A.T)),A_mm))
    '''
    A_rr = np.loadtxt("./Data/aBiofilm/Srr.txt")  # 药物节点属性表示矩阵 1373*256
    A_mm = np.loadtxt("./Data/aBiofilm/Smm.txt")  # 微生物节点属性表示矩阵 173*256
    Sr_che = np.loadtxt("./Data/attr embedding/attr embedding_Sr_che.txt")  # S_r^Che drug structure similarity
    Sm_fun = np.loadtxt("./Data/attr embedding/attr embedding_Sm_f.txt")  # S_m^f microbe function similarity
    Sr_m = np.loadtxt("./Data/attr embedding/attr embedding_Sr_dis.txt")
    Sm_r = np.loadtxt("./Data/attr embedding/attr embedding_Sm_dis.txt")
    #Srr = np.loadtxt("./Data/MDAD/Srr.txt")
    #Smm = np.loadtxt("./Data/MDAD/Smm.txt")

    # 将两个数组按水平方向组合起来
    Drug_feature = np.hstack((Sr_che, A))  # 1373*1802
    Drug_feature = np.hstack((np.hstack((Drug_feature, Sr_m)), A))  # 1373*3348
    Drug_feature = np.hstack((np.hstack((Drug_feature, A_rr)), A))
    Microbe_feature = np.hstack((A.T, Sm_fun))  # 173*1802
    Microbe_feature = np.hstack((np.hstack((Microbe_feature, A.T)), Sm_r))  # 173*3348
    Microbe_feature = np.hstack((np.hstack((Microbe_feature, A.T)), A_mm))
    '''
    return  Drug_feature,Microbe_feature

