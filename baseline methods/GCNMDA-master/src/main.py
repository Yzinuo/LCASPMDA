from utils import div_list
import tensorflow as tf
import numpy as np
from train import Training
import random
from sklearn.metrics import roc_curve,auc, precision_recall_curve,average_precision_score, recall_score, precision_score
known = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\MDAD\\known.txt")  # 已知关联索引（序号从1开始）
unknown = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\MDAD\\unknown.txt")
A = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\MDAD\\drug_microbe_matrix.txt")  # Adjacency matrx
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

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score

def accuracy(outputs, labels):
    assert len(labels.shape) == 1 and len(outputs.shape) == 1
    threshold_value = (tf.reduce_max(outputs) + tf.reduce_min(outputs)) / 2.
    outputs = tf.cast(tf.greater_equal(outputs, threshold_value), tf.bool)
    labels = tf.cast(labels, tf.bool)
    corrects = tf.math.logical_not(tf.math.logical_xor(outputs, labels))
    corrects = tf.cast(corrects, tf.int32)
    if tf.size(labels) == 0:
        return float('nan')
    return tf.reduce_sum(corrects) / tf.size(labels)


def f1_score(labels, predictions):
    labels = tf.cast(labels, tf.int32)
    predictions = tf.cast(predictions, tf.int32)

    # 计算TP, TN, FP, FN
    TP = tf.reduce_sum(predictions * labels)
    TN = tf.reduce_sum((1 - predictions) * (1 - labels))
    FP = tf.reduce_sum(predictions * (1 - labels))
    FN = tf.reduce_sum((1 - predictions) * labels)

    # 转换为浮点数
    TP = tf.cast(TP, tf.float32)
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)

    # 计算精确率和召回率
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1


# def mcc(outputs, labels):
#     assert len(labels.shape) == 1 and len(outputs.shape) == 1
#     outputs = tf.cast(tf.greater_equal(outputs, 0.5), tf.int32)
#     labels = tf.cast(labels, tf.int32)
#     true_pos = tf.reduce_sum(outputs * labels)
#     true_neg = tf.reduce_sum((1 - outputs) * (1 - labels))
#     false_pos = tf.reduce_sum(outputs * (1 - labels))
#     false_neg = tf.reduce_sum((1 - outputs) * labels)
#     numerator = tf.cast(true_pos * true_neg - false_pos * false_neg, tf.float32)
#     deno_2 = tf.reduce_sum(outputs) * tf.reduce_sum(1 - outputs) * tf.reduce_sum(labels) * tf.reduce_sum(1 - labels)
#     if deno_2 == 0:
#         return float('nan')
#     return numerator / tf.sqrt(tf.cast(deno_2, tf.float32))

def mcc(outputs, labels):
    assert len(labels.shape) == 1 and len(outputs.shape) == 1
    outputs = tf.cast(tf.greater_equal(outputs, 0.5), tf.int32)
    labels = tf.cast(labels, tf.int32)
    true_pos = tf.reduce_sum(outputs * labels)
    true_neg = tf.reduce_sum((1 - outputs) * (1 - labels))
    false_pos = tf.reduce_sum(outputs * (1 - labels))
    false_neg = tf.reduce_sum((1 - outputs) * labels)
    numerator = tf.cast(true_pos * true_neg - false_pos * false_neg, tf.float32)
    deno_1 = tf.cast(true_pos + false_pos, tf.float32) * tf.cast(true_pos + false_neg, tf.float32)
    deno_2 = tf.cast(true_neg + false_pos, tf.float32) * tf.cast(true_neg + false_neg, tf.float32)
    deno = tf.sqrt(deno_1 * deno_2)
    if deno == 0:
        return float('nan')
    return numerator / deno


if __name__ == "__main__":
  # Initial model
  gcn = Training()
  
  # Set random seed
  seed = 123
  np.random.seed(seed)
  tf.random.set_seed(seed)

  labels = np.loadtxt("E:\\实验室\\GCNMDA-master\\data\\MDAD\\adj.txt")
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

      endlabels = tf.convert_to_tensor(endlabels, dtype=tf.float32)
      scores_list = tf.convert_to_tensor(scores_list, dtype=tf.float32)

      num = str(cv_num)
      np.savetxt("E:\实验室\GCNMDA-master\data\\MDAD\\fpr_tpr\\fpr" + num + ".txt", fpr)
      np.savetxt("E:\实验室\GCNMDA-master\data\\MDAD\\fpr_tpr\\tpr" + num + ".txt", tpr)
      np.savetxt("E:\实验室\GCNMDA-master\data\\MDAD\\PR\\precision" + num + ".txt", precision)
      np.savetxt("E:\实验室\GCNMDA-master\data\\MDAD\\PR\\recall" + num + ".txt", recall)

      auc_val = auc(fpr, tpr)
      PR = auc(recall, precision)
      acc = accuracy(scores_list,endlabels)
      mcc = mcc(scores_list,endlabels)
      scores_new = tf.greater_equal(scores_list, 0.5)
      f1 = f1_score(endlabels, scores_new)

      sess = tf.compat.v1.Session()
      acc_value, f1_value, mcc_value = sess.run([ acc, f1, mcc])
      print("acc : {}".format(acc_value))
      print("f1 : {}".format(f1_value))
      print("mcc : {}".format(mcc_value))
      print('------------------------------------------------------')
      print(auc_val)
      print(PR)
      print('------------------------------------------------------')


 
