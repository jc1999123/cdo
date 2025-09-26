
from re import X

import numpy as np
from sklearn.datasets import make_s_curve
import torch
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


import torch.nn.functional as F
import math

import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n ** 2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n ** 2 / (n - 1) ** 2


def HSIC(x, y):
    Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    Kx = np.exp(- Kx ** 2)  # 计算核矩阵

    Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    Ky = np.exp(- Ky ** 2)  # 计算核矩阵
    return hsic(Kx, Ky)


def MSE(y, t):
    # 形参t代表训练数据（监督数据）（真实）
    # y代表预测数据
    return np.sum((y - t) ** 2) / y.shape[0]


def dis_hisc(y_pre, y_test):
    # y_test_pre = y_pre.detach().numpy()
    # finalloss=HSIC(y_test_pre,y_test)
    # MSE
    y_test_pre = y_pre.detach().numpy()
    y_test = y_test.detach().numpy()
    finalloss = MSE(y_test.reshape(-1), y_test_pre.reshape(-1))
    # y_test = torch.from_numpy(y_test)
    # finalloss= F.kl_div(y_pre.softmax(dim=-1).log(), y_test.softmax(dim=-1), reduction='sum')
    # finalloss = finalloss.detach().numpy()

    return finalloss


def changezereo(ff):
    for i in range(ff.shape[0]):
        for j in range(ff.shape[1]):
            if ff[i, j] < 0:
                ff[i, j] = 0
            if ff[i, j] > 0:
                ff[i, j] = 1

    return ff


def result_auc1(f1_result, f2_result, gold):
    f3 = f2_result[:, :, 0] - f1_result[:, :, 0]
    f3_1 = f2_result[:, :, 1] - f1_result[:, :, 1]
    f3_2 = f2_result[:, :, 2] - f1_result[:, :, 2]
    # f3_3=f2_result[:,:,3]-f1_result[:,:,3]
    # f3_4=f2_result[:,:,4]-f1_result[:,:,4]

    f_real = np.zeros(f3.shape)
    i = 0
    # while (gold[i,2]=='1'):
    #   print(1)

    for i in range(gold.shape[0]):
        if gold[i, 2] == '0':
            break
        else:
            q1 = gold[i, 0]
            q2 = gold[i, 1]
            q1 = int(q1[1:])
            q2 = int(q2[1:])
            f_real[q1 - 1, q2 - 1] = 1

    f5 = np.zeros(f3.shape)

    ##f5=f3+f3_1+f3_2
    f5 = f3 + f3_1 + f3_2
    # +f3_3+f3_4

    f3_1 = changezereo(f3_1)
    f3_2 = changezereo(f3_2)
    # ##f3_3=changezereo(f3_3)
    # ###f3_4=changezereo(f3_4)
    f3 = changezereo(f3)

    f4 = np.zeros(f3.shape)

    for i in range(f4.shape[0]):
        for j in range(f4.shape[1]):
            f4[i, j] = f3[i, j] + f3_1[i, j] + f3_2[i, j]
            # +f3_3[i,j]+f3_4[i,j]
            if f4[i, j] > 1:
                f4[i, j] = 1
            else:
                f4[i, j] = 0

            if f5[i, j] < 0:
                f5[i, j] = 0
            else:
                f5[i, j] = 1
                a = 1

    # print(f4)
    # print(f4)
    # print(f5)

    y_pre = f4.reshape(-1)
    y_test = f_real.reshape(-1)

    y_pre2 = f5.reshape(-1, 1)

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre)
    roc_auc = metrics.auc(fpr, tpr)
    precision1, recall1, _ = metrics.precision_recall_curve(y_test, y_pre)
    aupr1 = metrics.auc(recall1, precision1)  # the value of roc_auc1
    acc1 = accuracy_score(y_test, y_pre)
    print(aupr1, 'aupr')

    print(roc_auc, 'f4')
    print(acc1, 'acc2')

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre2)
    roc_auc = metrics.auc(fpr, tpr)
    precision2, recall2, _ = metrics.precision_recall_curve(y_test, y_pre2)
    aupr2 = metrics.auc(recall2, precision2)
    acc2 = accuracy_score(y_test, y_pre2)
    print(aupr2, 'aupr2')
    print(roc_auc, 'f5')
    print(acc2, 'acc2')


def result_auc2(f1_result, f2_result, gold):
    f3 = f1_result[:, :, 0] - f2_result[:, :, 0]
    f3_1 = f1_result[:, :, 1] - f2_result[:, :, 1]
    f3_2 = f1_result[:, :, 2] - f2_result[:, :, 2]
    # f3_3=f2_result[:,:,3]-f1_result[:,:,3]
    # f3_4=f2_result[:,:,4]-f1_result[:,:,4]

    f_real = np.zeros(f3.shape)
    i = 0
    # while (gold[i,2]=='1'):
    #   print(1)

    for i in range(gold.shape[0]):
        if gold[i, 2] == '0':
            break
        else:
            q1 = gold[i, 0]
            q2 = gold[i, 1]
            q1 = int(q1[1:])
            q2 = int(q2[1:])
            f_real[q1 - 1, q2 - 1] = 1

    f5 = np.zeros(f3.shape)

    ##f5=f3+f3_1+f3_2
    f5 = f3 + f3_1 + f3_2
    # +f3_3+f3_4

    f3_1 = changezereo(f3_1)
    f3_2 = changezereo(f3_2)
    # ##f3_3=changezereo(f3_3)
    # ###f3_4=changezereo(f3_4)
    f3 = changezereo(f3)

    f4 = np.zeros(f3.shape)

    for i in range(f4.shape[0]):
        for j in range(f4.shape[1]):
            f4[i, j] = f3[i, j] + f3_1[i, j] + f3_2[i, j]
            # +f3_3[i,j]+f3_4[i,j]
            if f4[i, j] > 1:
                f4[i, j] = 1
            else:
                f4[i, j] = 0

            if f5[i, j] < 0:
                f5[i, j] = 0
            else:
                f5[i, j] = 1
                a = 1

    # print(f4)
    # print(f4)
    # print(f5)

    y_pre = f4.reshape(-1)
    y_test = f_real.reshape(-1)

    y_pre2 = f5.reshape(-1, 1)

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre)
    roc_auc = metrics.auc(fpr, tpr)
    precision1, recall1, _ = metrics.precision_recall_curve(y_test, y_pre)
    aupr1 = metrics.auc(recall1, precision1)  # the value of roc_auc1
    acc1 = accuracy_score(y_test, y_pre)
    print(aupr1, 'aupr')

    print(roc_auc, 'f4')
    print(acc1, 'acc2')

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pre2)
    roc_auc = metrics.auc(fpr, tpr)
    precision2, recall2, _ = metrics.precision_recall_curve(y_test, y_pre2)
    aupr2 = metrics.auc(recall2, precision2)
    acc2 = accuracy_score(y_test, y_pre2)
    print(aupr2, 'aupr2')
    print(roc_auc, 'f5')
    print(acc2, 'acc2')

