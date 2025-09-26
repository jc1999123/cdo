import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import torch.nn.functional as F
import math
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def changezereo(ff):
    for i in range(ff.shape[0]):
        for j in range(ff.shape[1]):
            if ff[i, j] < 0:
                ff[i, j] = 0
            if ff[i, j] > 0:
                ff[i, j] = 1

    return ff


def result_auc(f1_result, f2_result, cishu,gold):
    f3 = f2_result[:, :, 0] - f1_result[:, :, 0]
    f3_1 = f2_result[:, :, 1] - f1_result[:, :, 1]
    f3_2 = f2_result[:, :, 2] - f1_result[:, :, 2]
    f3_3 = f2_result[:, :, 3] - f1_result[:, :, 3]
    f3_4 = f2_result[:, :, 4] - f1_result[:, :, 4]
    f3_5 = f2_result[:, :, 5] - f1_result[:, :, 5]
    f3_6 = f2_result[:, :, 6] - f1_result[:, :, 6]
    f3_7 = f2_result[:, :, 7] - f1_result[:, :, 7]
    f3_8 = f2_result[:, :, 8] - f1_result[:, :, 8]
    f3_9 = f2_result[:, :, 9] - f1_result[:, :, 9]
    # print(f3)

    ###jisuan真实网络
    f_real=np.zeros(f3.shape)
    # i=0
    # #while (gold[i,2]=='1'):
    # #   print(1)

    for i in range(gold.shape[0]):
        if gold[i,2] == '0':
            break
        else:
            q1=gold[i,0]
            q2=gold[i,1]
            q1=int(q1[1:])
            q2=int(q2[1:])
            f_real[q1-1,q2-1]=1

    f5 = np.zeros(f3.shape)
    f6 = np.zeros(f3.shape)
    f7 = np.zeros(f3.shape)

    ##f5=f3+f3_1+f3_2
    f5 = f3 + f3_1 + f3_2 + f3_4 + f3_5 + f3_6 + f3_7 + f3_8 + f3_9
    f6 = f3 + f3_1 + f3_2 + f3_3 + f3_4 + f3_5 + f3_6 + f3_7 + f3_8 + f3_9
    f3_1 = changezereo(f3_1)
    f3_2 = changezereo(f3_2)
    f3_3 = changezereo(f3_3)
    f3_4 = changezereo(f3_4)
    f3 = changezereo(f3)
    f3_5 = changezereo(f3_5)
    f3_6 = changezereo(f3_6)
    f3_7 = changezereo(f3_7)
    f3_8 = changezereo(f3_8)
    f3_9 = changezereo(f3_9)

    f4 = np.zeros(f3.shape)
    # print(f3.shape)

    for i in range(f4.shape[0]):
        for j in range(f4.shape[1]):
            f4[i, j] = f3[i, j] + f3_1[i, j] + f3_2[i, j] + f3_3[i, j] + f3_4[i, j] + f3_5[i, j] + f3_6[i, j] + f3_7[
                i, j] + f3_8[i, j] + f3_9[i, j]
            # f6[i,j] = f4[i,j]
            if f4[i, j] >= 2:
                f4[i, j] = 1
            else:
                f4[i, j] = 0

            if f5[i, j] <= 0:
                f5[i, j] = 0
            else:
                f5[i, j] = 1
                a = 1

    # print(f4)
    # print(f4)
    # print(f5)

    y_pre=f4.reshape(-1)
    y_test=f_real.reshape(-1)
    # print(f3)
    # f6 = f3+f3_1+f3_2+f3_3+f3_4+f3_5+f3_6+f3_7+f3_8+f3_9

    # print(f6)
    # exit()
    y_pre2=f5.reshape(-1,1)

    # pd.DataFrame(f4).to_csv('ukr1_result/4_' + str(cishu) + '.csv')
    # pd.DataFrame(f5).to_csv('ukr1_result/5_' + str(cishu) + '.csv')
    # pd.DataFrame(f6).to_csv('ukr1_result/6_' + str(cishu) + '.csv')
    # pd.DataFrame(f7).to_csv('school_result/f7_'+str(cishu)+'.csv')

    # '''
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre)
    roc_auc = metrics.auc(fpr, tpr)
    precision1, recall1, _ = metrics.precision_recall_curve(y_test, y_pre)
    aupr1 = metrics.auc(recall1, precision1)  # the value of roc_auc1
    acc1=accuracy_score(y_test, y_pre)
    print(aupr1,'aupr')
    print(roc_auc,'f4')
    print(acc1,'acc2')


    fpr2, tpr2, threshold2 = metrics.roc_curve(y_test, y_pre2)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    precision2, recall2, _ = metrics.precision_recall_curve(y_test, y_pre2)
    aupr2 = metrics.auc(recall2, precision2)
    acc2=accuracy_score(y_test, y_pre2)
    print(aupr2,'aupr2')
    print(roc_auc2,'f5')
    print(acc2,'acc2')
    # '''


nn = 10
cishu = 1
k_fold = 3
name_new = 4
f1_result = np.zeros((nn, nn, 10))
f2_result = np.zeros((nn, nn, 10))
# for cishu in range(cishu_count):
for i in range(k_fold):
    exe = pd.read_table('model/clu1' + str(i) + 'mse0.001_adam.csv', sep=',')
    exe2 = pd.read_table('model/clu1' + str(i) + 'mse0.001_adam.csv', sep=',')
    f1_result[:, :, i] = np.array(exe.iloc[:, 1:])
    f2_result[:, :, i] = np.array(exe2.iloc[:, 1:])

gold=np.loadtxt('data/insilico_size10_'+str(name_new)+'_goldstandard.tsv',dtype='str')
result_auc(f1_result, f2_result, cishu,gold)




