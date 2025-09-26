import numpy as np
import torch
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
from difftcnode import DiffTCNODE  # Import the model from model.py
from dataread import read_data,generate_dataset
import torch.nn as nn
from tqdm import tqdm
from args import args
from causaldif import cau_model

from decesion import dis_hisc

from do_result import pre_A ,pre_A_1

from sklearn.model_selection import TimeSeriesSplit
import numpy as np


def delzero(expression,threshold = 0.8 ):  # 设置阈值为 80%
    # 计算每列中 0 所占的比例
    zero_ratio = (expression == 0).mean()

    # 只保留 0 比例小于 80% 的列
    df_filtered = expression.loc[:, zero_ratio < threshold]
    return df_filtered


def diffusion_loss_fn(model, x_0, targets, criterion ,alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps,batch_size):

    # batch_size = x_0.shape[0]
    # choose = torch.randint(0, 4,size=(1,1))
    choose = 0
    if choose < 1 :
        t = torch.randint(0, int(n_steps/4), size=(batch_size // 2,))
        t = torch.cat([t, n_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)

        a = alphas_bar_sqrt[t]
        # a = a.unsqueeze(-1)
        # a = a.unsqueeze(-1)
        a = a.unsqueeze(-1).unsqueeze(-1)
        # xx = x_0 * a
        aml = one_minus_alphas_bar_sqrt[t]
        aml = aml.unsqueeze(-1).unsqueeze(-1)
        e = torch.randn_like(x_0)
        # print(x_0.shape,'x_0.shape')
        # print(a.shape,'a.shape')
        # print(e.shape, 'e.shape')
        # print(aml.shape, 'aml.shape')
        x = x_0 * a + e * aml

        x_t, output = model(x, t.squeeze(-1),choose)
        e= e[:, :, :, :1]
        e = e.squeeze()
        # print(x_t.shape,'x_t.shape')
        # print(targets.shape,'target')
        loss2 = criterion(x_t, targets)
        loss1 = (e - output).square().mean()
        loss = loss2
        control = loss1
        return loss ,control

    else:
        t = torch.randint(int(n_steps / 4), n_steps, size=(batch_size // 2,))
        t = torch.cat([t, n_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)

        a = alphas_bar_sqrt[t]
        # a = a.unsqueeze(-1)
        # a = a.unsqueeze(-1)
        a = a.unsqueeze(-1).unsqueeze(-1)
        # xx = x_0 * a
        aml = one_minus_alphas_bar_sqrt[t]
        aml = aml.unsqueeze(-1).unsqueeze(-1)
        e = torch.randn_like(x_0)
        # print(x_0.shape,'x_0.shape')
        # print(a.shape,'a.shape')
        # print(e.shape, 'e.shape')
        # print(aml.shape, 'aml.shape')
        x = x_0 * a + e * aml

        output = model(x, t.squeeze(-1),choose)

        e= e[:, :, :, :1]
        e = e.squeeze()
        control = 10
        loss = (e - output).square().mean()
        return loss ,control

#
# expression = pd.read_table('data/insilico_size10_1_timeseries.tsv', sep='\t')
# expression = expression.iloc[0:100, 1:]

# expression = pd.read_table('data/unobserved/unobserved_nolinera_3_ode_2.csv', sep='\t')
# # expression = np.loadtxt('data/sim6_gold.txt', dtype='str')
# print(expression)
# expression = pd.read_table('data/sim6.csv', sep=',',header=None)
# expression = np.loadtxt('data/sim6_gold.txt', dtype='str')
# print(expression)
# exit()

'''
####durg expression
expression = np.load('data/drug3all/all_layer_drug3.npy')

####Drug取前三种药物

# print(expression)
expression =pd.DataFrame(expression)
# print(expression)
# expression =expression.iloc[:,0:3]
# expression.columns=['G1','G2','G3']
expression =expression.iloc[:,0:5]
expression.columns=['G1','G2','G3','G4','G5']
'''








# expression = pd.read_table('data/drug3all/allmydata.csv',sep=',')

#     ####Drug取前三种药物

# # print(expression)
# expression =pd.DataFrame(expression)
# print(expression.shape)
# expression =expression.iloc[0:100,:]
# expression =delzero(expression)

# print(expression.shape)




expression = np.load('data/causaltime/causaltime_gen_ver1.0/traffic/gen_data.npy')
print(expression.shape)
# expression = np.load('data/causaltime/causaltime_gen_ver1.0/medical/graph.npy')
# print(expression)
# expression =expression.reshape(480*40, 40)
expression =expression.reshape(480*40, 40)
# expression =expression[1]

# pd.read_table('data/sim6.csv', sep=',',header=None)
# print(expression)
expression =pd.DataFrame(expression)
print(expression.shape)
expression =expression.iloc[80:120,0:20]






# expression =expression.iloc[:,1:11]
# print(expression)
# # expression.columns=['G1','G2','G3','G4','G5']
# expression.columns = [f'G{i}' for i in range(1, 11)]
# print(expression)






A = np.zeros((expression.shape[1],expression.shape[1]))
# expression = expression.iloc[1200:2400, :]
# expression =np.zeros((3,3))
# print(expression)
A = np.array(expression.corr())
# A = (A > 0.3).astype(int)
A = (abs(A) > 0).astype(int)
A_sp_hat = torch.from_numpy(A.astype(np.float32))
# print(A)
# exit()

nn1 = expression.shape[1]
cishu = 1
k_fold = 3
f1_result = np.zeros((nn1,nn1,10))
f2_result = np.zeros((nn1,nn1,10))

#
# for i in range(k_fold):
#     exe = pd.read_table('model/dream4/4_5/test1/clu1'+ str(kk)  + str(i) + "_"+str(loop)+'mse0.001_adam.csv',sep=',')
#     exe2 = pd.read_table('model/dream4/4_5/test1/clu1'+ str(kk)  + str(i) + "_"+str(loop)+'mse0.001_adam.csv',sep=',')
#     f1_result[:,:,i] = np.array(exe.iloc[:,1:])
#     f2_result[:,:,i] = np.array(exe2.iloc[:,1:])
#
# gold = np.loadtxt('data/insilico_size10_4_goldstandard.tsv', dtype='str')
#
#
#
#
# A_sp_hat = pre_A_1(f1_result, f2_result,gold)




num_nodes = expression.shape[1]
num_features = 1  ###默认是1
num_timesteps_input = 4
num_timesteps_output = 4
n_steps = 200
num_steps =200

model = DiffTCNODE(num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat,n_steps)

kk1 = 1
loop1 = 0

loss_count =np.zeros((3,1))
for loop in range(19):
    for kk in range(3):
        loop =loop1
        # kk = kk1
    # model_path = "model/dream4/4_1/tcntest1/clu1_" + str(kk) + "_" + str(loop) + "_3model.h5"
    # model_path = "model/nolinera_ode/tcn_unobserved_case2_1/clu1_" + str(kk) + "_" + str(loop) + "_3model.h5"
        model_path = "model/causaltime/clu1_" + str(kk) + "_" + str(loop) + "_3model.h5"
        model.load_state_dict(torch.load(model_path))

        x_now = np.array(expression.values)
        all_train = x_now
        data = read_data(all_train)
        train_loader, valid_loader, test_loader = generate_dataset(data, args)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        # print(alphas_bar_sqrt)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        model.eval()
        epoch_loss = 0

        num_batches = len(train_loader)
        for idx, (inputs, targets) in enumerate(train_loader):
            # model.eval()
            loss, control = diffusion_loss_fn(model, inputs, targets, criterion, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                            num_steps, args.batch_size)
            epoch_loss += loss.item()

        average_loss = epoch_loss / num_batches
        loss_count[kk] = average_loss
        print(average_loss,'')
print(sum(loss_count)/20,'loss_avr')

# loop = 20
# for i in range(loop):
#     A = train_model(expression, A, args, i)