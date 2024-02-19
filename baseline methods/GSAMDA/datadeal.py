import numpy as np
import math




def HIP_Calculate(M):
    l=len(M)
    cl=np.size(M,axis=1)
    SM=np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M[i][k]!=M[j][k]:
                    dnum=dnum+1
            SM[i][j]=1-dnum/cl             #HIP计算出来的相似矩阵
    return SM
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
def Cosine_Sim(M):
    l=len(M)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            v1=np.dot(M[i],M[j])
            v2=np.linalg.norm(M[i],ord=2)
            v3=np.linalg.norm(M[j],ord=2)
            if v2*v3==0:
                SM[i][j]=0
            else:
                SM[i][j]=v1/(v2*v3)
    return SM


