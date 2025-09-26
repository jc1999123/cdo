import numpy as np
import torch
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
from difftcnode import DiffTCNODE  # Import the model from model.py
from dataread import read_data,generate_dataset,generate_dataset_test
import torch.nn as nn
from tqdm import tqdm
from args import args
from causaldif import cau_model
import os
from decesion import dis_hisc

from do_result import pre_A ,pre_A_1

from sklearn.model_selection import TimeSeriesSplit
import numpy as np

dataset = "sim"   # 改成 "B" 就能自动切换
# ======================================

# 统一路径
data_path = f"data/{dataset}.csv"
model_dir = f"model/{dataset}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# exit()

def diffusion_loss_fn(model, x_0, targets, criterion,
                      alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                      n_steps, batch_size):

    # 把数据移到 device
    x_0 = x_0.to(device)
    targets = targets.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

    choose = torch.randint(0, 4, size=(1,1), device=device)
    # print(x_0.shape)
    if choose < 1:

        t = torch.randint(0, int(n_steps/4), size=(batch_size // 2,), device=device)
        t = torch.cat([t, n_steps - 1 - t], dim=0).unsqueeze(-1)

        a = alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1)
        aml = one_minus_alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1)

        e = torch.randn_like(x_0)
        x = x_0 * a + e * aml
        ###确保targets与x_0同维度
        targets = targets.unsqueeze(-1)
        e1 = torch.randn_like(targets)
        targets = targets * a + e1 * aml
        targets = targets.squeeze(-1)

        x_t, output = model(x, t.squeeze(-1), choose)
        e = e[:, :, :, :1].squeeze()

        loss2 = criterion(x_t, targets)
        loss1 = (e - output).square().mean()
        loss = loss1 + loss2
        control = loss1

        return loss, control

    else:
        t = torch.randint(int(n_steps / 4), n_steps, size=(batch_size // 2,), device=device)
        t = torch.cat([t, n_steps - 1 - t], dim=0).unsqueeze(-1)

        a = alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1)
        aml = one_minus_alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1)

        e = torch.randn_like(x_0)
        x = x_0 * a + e * aml

        output = model(x, t.squeeze(-1), choose)
        e = e[:, :, :, :1].squeeze()

        loss = (e - output).square().mean()
        control = 10
        return loss, control



def diffusion_loss_fn_test(model, x_0, targets, criterion ,alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps,batch_size):

    # batch_size = x_0.shape[0]
    choose = 0
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
    loss = loss1 + loss2
    control = loss2
    return loss ,control


def diffusion_loss_fn_test_gpu(
    model, x_0, targets, criterion,
    alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
    n_steps, batch_size, device
):
    # 确保输入在 GPU
    x_0 = x_0.to(device)
    # print(x_0.shape)
    targets = targets.to(device)

    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

    # 生成时间步，放到 GPU
    # t = torch.randint(0, int(n_steps/4), size=(batch_size,), dtype=torch.long, device=device)
    t = torch.randint(0, int(n_steps / 4), size=(batch_size // 2,), device=device)

    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)
    # choose 变量如果需要，也放到 GPU
    choose = 0

    # 计算噪声
    noise = torch.randn_like(x_0, device=device)

    # 扩展系数到 batch 维度
    a = alphas_bar_sqrt[t].view(-1, 1, 1, 1).to(device)
    aml = one_minus_alphas_bar_sqrt[t].view(-1, 1, 1, 1).to(device)
    # print(x_0.shape, 'x')
    x = a * x_0 + aml * noise

    # 前向传播
    x_t, output = model(x, t.squeeze(-1), choose)
    e= noise[:, :, :, :1]
    e = e.squeeze()
    # print(x_t.shape,'x_t.shape')
    # print(targets.shape,'target')

    ###确保targets与x_0同维度
    targets = targets.unsqueeze(-1)
    e1 = torch.randn_like(targets)
    targets = targets * a + e1 * aml
    targets = targets.squeeze(-1)


    loss2 = criterion(x_t, targets)
    loss1 = (e - output).square().mean()
    loss = loss1 + loss2
    control = loss2
    # print(x.shape,'x')
    # print(x_t.shape,'x_t',targets.shape,'target')
    # loss
    # loss = criterion(output, targets)
    # loss2 = criterion(x_t, targets)
    # loss1 = (e - output).square().mean()
    # loss = loss1 + loss2
    # loss= 0
    # control = 0
    # control = criterion(x_t, targets)
    return loss, control
    
def train_model(expression, A_sp_hat, args, loop, num_steps=200, num_epoch=200,
                batch_size=64, k_fold=3):

    num_nodes = expression.shape[1]
    num_features = 1
    num_timesteps_input = 4
    num_timesteps_output = 4
    n_steps = num_steps

    betas = torch.linspace(-6, 6, num_steps, device= device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    x_now = np.array(expression.values)
    tscv = TimeSeriesSplit(n_splits=k_fold)

    kk = 0
    f1_result = np.zeros((expression.shape[1], expression.shape[1], k_fold))
    f2_result = np.zeros((expression.shape[1], expression.shape[1], k_fold))

    for train_index, test_index in tscv.split(x_now):

        all_train, all_test = x_now[train_index], x_now[test_index]

        if all_train.shape[0] % args.batch_size != 0:
            y = (all_train.shape[0] // args.batch_size) * args.batch_size
            print()
            all_train = all_train[:y]

        data = read_data(all_train)
        train_loader, _, _ = generate_dataset(data, args)

        # 初始化模型到 GPU
        A_sp_hat = A_sp_hat.to(device)
        model = DiffTCNODE(num_nodes, num_features, num_timesteps_input,
                           num_timesteps_output, A_sp_hat, n_steps).to(device)

        if loop != 0:
            model_path = os.path.join(model_dir, f"clu1_{kk}_{loop-1}_3model.h5")
            model.load_state_dict(torch.load(model_path, map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.SmoothL1Loss()

        model.train()
        for t in range(num_epoch):
            epoch_loss = 0.0
            num_samples = 0

            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                loss, control = diffusion_loss_fn(
                    model, inputs, targets, criterion,
                    alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                    num_steps, args.batch_size
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                bs = inputs.size(0)
                epoch_loss += loss.item() * bs
                num_samples += bs

            avg_loss = epoch_loss / max(1, num_samples)

        # 保存模型
        save_path = os.path.join(model_dir, f"clu1_{kk}_{loop}_3model.h5")
        torch.save(model.state_dict(), save_path)



        model.eval()
        ####所有参数再放回CPU进行运算
        # exit()

        # device1 = torch.device("cpu")
        # model = model.to(device1)
        #
        # alphas_bar_sqrt = alphas_bar_sqrt.to(device1)
        # one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device1)
        #
        # # 如果 betas 在 GPU，也要转
        # betas = betas.to(device1)
        #
        # # 如果 inputs / targets 已经提前加载，也转到 CPU
        # inputs = inputs.to(device1)
        # targets = targets.to(device1)


        loss,control = diffusion_loss_fn_test_gpu(model, inputs,targets, criterion, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,args.batch_size,device)
        print(control,'shixuloss','in',loop,'loop')
        # model.state_dict()
        save_path = os.path.join(model_dir, f"clu1_{kk}_{loop}_3model.h5")
        torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(), "model/alldrugtest/alldata_case1/clu1_" + str(kk) + "_"+str(loop)+"_3model.h5")
        # exit()
        # test_dataset
        # exit()

        test_dataset = []
        test_dataset_labels = []

        all_test = read_data(all_test)
        test_loader, valid_loader_non, test_loader_non = generate_dataset_test(all_test, args)
        # for idxqq, (inputs, targets) in enumerate(test_loader):
        # exit()
        for i in range(expression.shape[1]):
            for idxqq, (inputs, targets) in enumerate(test_loader):
                # print(inputs.shape)
                if idxqq !=0:
                    break
                # print(idxqq, 'idxqq')
                # print(inputs.shape,'inputshape')
                final1, final2 = cau_model(model, inputs.shape, num_steps, betas, one_minus_alphas_bar_sqrt, alphas_bar_sqrt,i,
                                            inputs)
                break
                # print(idxqq,'idxqq')
            ###计算距离
            for j in range(expression.shape[1]):
                if j == i:
                    continue
                else:
                    # print(inputs.shape)
                    # exit()
                    f1_result[i, j, kk] = dis_hisc(final1[:, j,:], inputs[:, j,:,:])
                    f2_result[i, j, kk] = dis_hisc(final2[:, j,:], inputs[:, j,:,:])

        # exit()
        kk += 1
    for i in range(k_fold):
        f1_path = os.path.join(model_dir, f"f1clu1_{kk}_{i}_{loop}_mse0.001_adam.csv")
        f2_path = os.path.join(model_dir, f"f2clu1_{kk}_{i}_{loop}_mse0.001_adam.csv")
        # pd.DataFrame(f1_result[:, :, i]).to_csv('model/alldrugtest/alldata_case1/f1clu1'+ str(kk)  + str(i) + "_"+str(loop)+'mse0.001_adam.csv')
        # pd.DataFrame(f2_result[:, :, i]).to_csv('model/alldrugtest/alldata_case1/f2clu1' + str(kk) + str(i) + "_"+str(loop)+'mse0.001_adam.csv')
        pd.DataFrame(f1_result[:, :, i]).to_csv(f1_path)
        pd.DataFrame(f2_result[:, :, i]).to_csv(f2_path)

    print(loop, 'loop finised')
    # gold = np.loadtxt('data/insilico_size10_4_goldstandard.tsv', dtype='str')
    # gold = np.loadtxt('data/unobserved/unobserved_ode_3_1_gold_2.txt', dtype='str')
    # gold = np.load('data/causaltime/causaltime_gen_ver1.0/medical/graph.npy')
    gold =np.loadtxt('data/sim/sim1_gold.txt',dtype ='str')
    A_next = pre_A_1(f1_result, f2_result, gold)

    return A_next

def delzero(expression,threshold = 0.8 ):  # 设置阈值为 80%
    # 计算每列中 0 所占的比例
    zero_ratio = (expression == 0).mean()

    # 只保留 0 比例小于 80% 的列
    df_filtered = expression.loc[:, zero_ratio < threshold]
    return df_filtered



if __name__ == "__main__":

    # expression = pd.read_table('data/insilico_size10_4_timeseries.tsv', sep='\t')
    # expression = expression.iloc[0:100, 1:]
    # expression = pd.read_table('data/sim6.csv', sep=',',header=None)
    # expression = pd.read_table('data/unobserved/unobserved_nolinera_3_ode_4.csv', sep='\t')
    # expression = np.loadtxt('data/sim6_gold.txt', dtype='str')
    # expression = np.load('data/drug3all/all_layer_drug3.npy')
    # expression = pd.read_table('data/drug3all/allmydata.csv',sep=',')
    expression = pd.read_table('data/sim/sim1.csv', sep=',',header=None)
    ####Drug取前三种药物
    # expression = np.load('data/causaltime/causaltime_gen_ver1.0/medical/gen_data.npy')
    print(expression.shape)
    # expression = np.load('data/causaltime/causaltime_gen_ver1.0/medical/graph.npy')
    # print(expression)
    # expression =expression.reshape(480*40, 40)
    expression = expression.iloc[1:201,:]
    ####  traffic data
    # expression = expression[:, :, :20]        # shape: (480, 40, 20)
    # expression = expression.reshape(-1, 20)  # shape: (480*40, 20)
    ####

    # expression =expression.reshape(480*40, 40)
    # expression =expression[1]

    # pd.read_table('data/sim6.csv', sep=',',header=None)
    # print(expression)
    expression =pd.DataFrame(expression)
    print(expression.shape)
    # expression =expression.iloc[0:480,:]
    # expression =delzero(expression)
    # expression =expression.reshape(40,40)
    # print(expression.shape)
    print(expression.shape)

    # exit()


    # expression =expression.iloc[:,1:11]
    # print(expression)
    # expression.columns=['G1','G2','G3','G4','G5']
    # expression.columns = [f'G{i}' for i in range(1, 11)]
    # print(expression)

    # exit()
    A = np.zeros((expression.shape[1],expression.shape[1]))
    # expression = expression.iloc[1200:2400, :]
    # expression =np.zeros((3,3))
    # print(expression)
    A = np.array(expression.corr())
    # A = (A > 0.3).astype(int)
    A = (abs(A) > 0).astype(int)
    A= torch.from_numpy(A.astype(np.float32))
    # print(A)
    # exit()
    loop =20
    for i in range(loop):

        A = train_model(expression, A, args,i)

    # expression =pd.DataFrame(expression)
    # print("shape of s:", np.shape(expression))
    # exit()

    # train_model(expression,A,args)


'''
###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###test_loader中的bathsize记得改



'''