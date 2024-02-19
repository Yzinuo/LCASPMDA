from utils import div_list
import tensorflow as tf
import numpy as np
from train import Training
import random
from sklearn.metrics import roc_curve,auc, precision_recall_curve
known = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\aBiofilm\\known.txt")  # 已知关联索引（序号从1开始）
unknown = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\aBiofilm\\unknown.txt")
A = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\aBiofilm\\drug_microbe_matrix.txt")  # Adjacency matrx
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


scores_list = []
endlabels=[]

if __name__ == "__main__":
  # Initial model
  gcn = Training()
  
  # Set random seed
  seed = 123
  np.random.seed(seed)
  tf.random.set_seed(seed)

  labels = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\aBiofilm\\adj.txt")
  reorder = np.arange(labels.shape[0])
  np.random.shuffle(reorder)

  cv_num=5

  order = div_list(reorder.tolist(),cv_num)
  for i in range(cv_num):
      print("cross_validation:", '%01d' % (i))
      test_arr = order[i]
      arr = list(set(reorder).difference(set(test_arr)))
      np.random.shuffle(arr)
      train_arr = arr
      scores = gcn.train(train_arr, test_arr)

      if cv_num< 5:
          B1_indices = k[(cv_num- 1) * (lk // 5):(lk // 5) * cv_num]
          B1 = known[B1_indices, :]

          B2_indices = unk[(cv_num - 1) * len(B1_indices):(cv_num - 1) * len(B1_indices) + len(B1_indices)]
          B2 = unknown[B2_indices, :]
      else:
          B1_indices = k[(cv_num - 1) * (lk // 5):]
          B1 = known[B1_indices, :]

          B2_indices = unk[(cv_num - 1) * len(B1_indices):(cv_num - 1) * len(B1_indices) + len(B1_indices)]
          B2 = unknown[B2_indices, :]

      for i in range(len(B1)):
          index1 = int(B1[i, 0] - 1)
          index2 = int(B1[i, 1] - 1)
          scores_list.append(scores[index1, index2])
          endlabels.append(A[index1, index2])
      for i in range(len(B2)):
          index1 = int(B2[i, 0] - 1)
          index2 = int(B2[i, 1] - 1)
          scores_list.append(scores[index1, index2])
          endlabels.append(A[index1, index2])

      fpr, tpr, threshold = roc_curve(endlabels,  scores_list)
      precision, recall, _ = precision_recall_curve(endlabels, scores_list)
      num = str(cv_num)
      np.savetxt("E:\实验室\GCNMDA-master\data\\aBiofilm\\fpr_tpr\\fpr" + num + ".txt", fpr)
      np.savetxt("E:\实验室\GCNMDA-master\data\\aBiofilm\\fpr_tpr\\tpr" + num + ".txt", tpr)
      np.savetxt("E:\实验室\GCNMDA-master\data\\aBiofilm\\PR\\precision" + num + ".txt", precision)
      np.savetxt("E:\实验室\GCNMDA-master\data\\aBiofilm\\PR\\recall" + num + ".txt", recall)

      auc_val = auc(fpr, tpr)
      PR = auc(recall, precision)
      print(auc_val)
      print(PR)
      print('------------------------------------------------------')


 
