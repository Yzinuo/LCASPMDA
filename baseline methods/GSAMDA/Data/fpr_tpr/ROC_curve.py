import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
plt.switch_backend('Tkagg')

#取最好的auc值画ROC曲线
a=np.loadtxt("auc.txt")
b=dict()
for i in range(len(a)):
    b[i]=a[i]
num=[k for k,v in b.items() if v==max(a)]
maxindex=num[0]

fpr=np.loadtxt("fpr"+str(maxindex)+".txt")
tpr=np.loadtxt("tpr"+str(maxindex)+".txt")
auc_val=auc(fpr,tpr)
print(auc_val)
plt.figure()
plt.plot(fpr,tpr)
plt.show()