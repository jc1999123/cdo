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
'''

def p_sample_loop(model ,shape ,n_steps ,betas ,one_minus_alphas_bar_sqrt ,cur_x):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    # cur_x = torch.randn(shape)
    x_seq = [cur_x]
    # print(x_seq)
    # print(len(x_seq))
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model ,cur_x ,i ,betas ,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
        # print(len(x_seq))
    return x_seq

def p_sample(model ,x ,t ,betas ,one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t])
    x1=x
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    x = x.unsqueeze(-1).unsqueeze(-1)

    eps_theta = model(x ,t, choose =1)
    x = x.squeeze()
    mean = ( 1 /( 1 -betas[t]).sqrt() ) *( x -(coeff *eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)

def cau_model(model ,shape ,num_steps ,betas ,one_minus_alphas_bar_sqrt ,alphas_bar_sqrt,i ,all_test):

    cur_xx = torch.randn(shape)
    x_seq_noise =p_sample_loop(model ,shape ,num_steps ,betas ,one_minus_alphas_bar_sqrt ,cur_xx)
    x_seq_noise_final =x_seq_noise[num_steps]



    # cur =torch.from_numpy(all_test[:, i,:,:])
    cur_xx[:, i,:,:] = all_test[:, i,:,:]
    t_shape = cur_xx.shape[0]
    t=  torch.full((t_shape,), num_steps-1)
    e = torch.randn_like(cur_xx[:, i,:,:])
    a= alphas_bar_sqrt[t]
    a  = a.unsqueeze(-1).unsqueeze(-1)
    aml = one_minus_alphas_bar_sqrt[t]
    aml = aml.unsqueeze(-1).unsqueeze(-1)
    cur_xx[:, i,:,:] = cur_xx[:, i,:,:]* a + e * aml
    # cur_xx[:, i,:,:] = cur_xx[:, i,:,:]* alphas_bar_sqrt[t]+ e*one_minus_alphas_bar_sqrt[t]

    # e = torch.randn_like(cur_xx[:, i,:,:])
    # print('cur_xx[:, i,:,:]',cur_xx[:, i,:,:].shape)
    # print('alphas_bar_sqrt[num_steps]',alphas_bar_sqrt[t].shape)
    # print('one_minus_alphas_bar_sqrt[num_steps]',one_minus_alphas_bar_sqrt[t].shape)
    # exit()
    # exit()
    # cur_xx[:, i,:,:] = cur_xx[:, i,:,:]* alphas_bar_sqrt[num_steps]+ e*one_minus_alphas_bar_sqrt[num_steps]


    # print(cur_xx.shape)
    # print(x_seq_noise_final.shape,'x_seq_noise_final.shape')
    # print(x_seq_noise.shape,'x_seq_noise')
    # exit()

    x_seq_cau = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, cur_xx)
    x_seq_cau_final = x_seq_cau[num_steps]
    return x_seq_cau_final, x_seq_noise_final
'''

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, cur_x, device):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = cur_x.to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt, device)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, device):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t], device=device)
    x = x.to(device)

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    x = x.unsqueeze(-1).unsqueeze(-1)

    eps_theta = model(x, t, choose=1)
    x = x.squeeze()
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x, device=device)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z
    return sample

def cau_model(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, alphas_bar_sqrt, i, all_test, device="cuda"):
    """因果扩散采样：GPU计算，最后结果转回CPU"""

    # 模型放到GPU
    model = model.to(device)

    # 先在GPU上生成噪声
    cur_xx = torch.randn(shape, device=device)

    # 初始噪声扩散
    x_seq_noise = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, cur_xx, device)
    x_seq_noise_final = x_seq_noise[num_steps]

    # 替换第i个变量
    cur_xx[:, i, :, :] = all_test[:, i, :, :].to(device)

    t_shape = cur_xx.shape[0]
    t = torch.full((t_shape,), num_steps - 1, device=device)

    e = torch.randn_like(cur_xx[:, i, :, :], device=device)
    a = alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1).to(device)
    aml = one_minus_alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1).to(device)

    cur_xx[:, i, :, :] = cur_xx[:, i, :, :] * a + e * aml

    # 因果扩散采样
    x_seq_cau = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, cur_xx, device)
    x_seq_cau_final = x_seq_cau[num_steps]

    # 最后结果放回CPU
    return x_seq_cau_final.to("cpu"), x_seq_noise_final.to("cpu")