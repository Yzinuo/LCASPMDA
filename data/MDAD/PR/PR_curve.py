import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Tkagg')
from sklearn.metrics import roc_curve,auc
# #取最好的auc值画ROC曲线
# a=np.loadtxt("auc.txt")
# b=dict()
# for i in range(len(a)):

#     b[i]=a[i]
# num=[k for k,v in b.items() if v==max(a)]
# maxindex=num[0]

precision=np.loadtxt("precision"+str(1)+".txt")
recall=np.loadtxt("recall"+str(1)+".txt")
auc_val=auc(recall,precision)
print(auc_val)

plt.figure()
plt.plot(recall, precision, label='PR Curve')
plt.show()