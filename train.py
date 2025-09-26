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

dataset = "causaltime"   # 改成 "B" 就能自动切换
# ======================================

# 统一路径
data_path = f"data/{dataset}.csv"
model_dir = f"model/{dataset}"




def diffusion_loss_fn(model, x_0, targets, criterion ,alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps,batch_size):

    # batch_size = x_0.shape[0]
    choose = torch.randint(0, 4,size=(1,1))
    # choose =0
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
        e1 = torch.randn_like(targets)
        targets = targets * a + e1*aml

        x_t, output = model(x, t.squeeze(-1),choose)
        e= e[:, :, :, :1]
        e = e.squeeze()
        # print(x_t.shape,'x_t.shape')
        # print(targets.shape,'target')
        loss2 = criterion(x_t, targets)
        loss1 = (e - output).square().mean()
        loss = loss1 + loss2
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




# def diffusion_loss_fn(model, x_0, targets, criterion,
#                       alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
#                       t, choose):
#     """
#     计算单个时间步 t 的 loss
#     """
#     a = alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1)
#     aml = one_minus_alphas_bar_sqrt[t].unsqueeze(-1).unsqueeze(-1)
#     e = torch.randn_like(x_0)

#     x = x_0 * a + e * aml

#     if choose < 1:
#         # 针对 targets 做相同扰动
#         e1 = torch.randn_like(targets)
#         targets = targets * a + e1 * aml

#         x_t, output = model(x, t.squeeze(-1), choose)

#         e = e[:, :, :, :1].squeeze()
#         loss2 = criterion(x_t, targets)
#         loss1 = (e - output).square().mean()
#         loss = loss1 + loss2
#         control = loss1
#     else:
#         output = model(x, t.squeeze(-1), choose)
#         e = e[:, :, :, :1].squeeze()
#         loss = (e - output).square().mean()
#         control = 10

#     return loss, control



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

    
    # t = torch.randint(0, n_steps, size=(batch_size // 2,))
    # t = torch.cat([t, n_steps - 1 - t], dim=0)
    # t = t.unsqueeze(-1)
    #
    # a = alphas_bar_sqrt[t]
    # a= a.unsqueeze(-1)
    # a =a.unsqueeze(-1)
    # xx =x_0*a
    # aml = one_minus_alphas_bar_sqrt[t]
    # aml =aml.unsqueeze(2).unsqueeze(3)
    # e = torch.randn_like(x_0)
    # x = x_0 * a + e * aml
    # if
    # if t < (n_steps/4):
    #
    #     x_t ,output = model(x, t.squeeze(-1))
    #
    #     loss1 = criterion(x_t, targets)
    #     loss2 = (e - output).square().mean()
    #
    #     return loss1+loss2
    #
    # else:
    #
    #     output = model(x, t.squeeze(-1))
    #
    #     return (e - output).square().mean()



def train_model(expression,A_sp_hat,args, loop ,num_steps=200, num_epoch=2000, batch_size=64, k_fold=3):

    ### arg
    num_nodes = expression.shape[1]
    num_features = 1###默认是1
    num_timesteps_input = 4
    num_timesteps_output = 4
    n_steps = num_steps


    # data = read_data(expression)
    # train_loader, valid_loader, test_loader = generate_dataset(data, args)
    # dataset = torch.Tensor(expression.values).float()
    #
    # dataloader = DataLoader(dataset, batch_size=batch_size)

    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    # print(alphas_bar_sqrt)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    # print(one_minus_alphas_bar_sqrt,'print(one_minus_alphas_bar_sqrt)')
    # exit()


    x_now = np.array(expression.values)
    # kf = KFold(n_splits=k_fold)
    kk = 0
    f1_result = np.zeros((expression.shape[1], expression.shape[1], k_fold))
    f2_result = np.zeros((expression.shape[1], expression.shape[1], k_fold))



    # 示例时序数据
    # data = np.arange(10)

    # 创建 TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=k_fold)

    # 打印每一折的训练集和测试集
    for train_index, test_index in tscv.split(x_now):
        # print("Train:", x_now[train_index].shape, "Test:", x_now[test_index].shape)

    ###不能用cross_vali 时序数据不能打乱
    # for train_index, test_index in kf.split(x_now):

        all_train, all_test = x_now[train_index], x_now[test_index]
        # dataset_train = torch.Tensor(all_train).float()
        # print(all_train[1:4],'\t','all_train','\t')
        # all_train=x_now
        # print(x_now.shape,'xnowshape')
        # print(all_train.shape,'alltrainshape')
        # print(all_test.shape,'alltestshape')
        if all_train.shape[0] % args.batch_size != 0:
            # 计算新的第一维度 y
            y = (all_train.shape[0] // args.batch_size) * args.batch_size  # 向下取整为 4 的倍数
            all_train = all_train[:y]
            # print("Train_new:", all_train.shape)
        data = read_data(all_train)
        # print(all_train[1:4])
        train_loader, valid_loader_non, test_loader_non  = generate_dataset(data, args)


        # print(test_dataset.shape ,'a')
        # exit()
        # print(data.shape,'datashape')
        # print(data)


        model = DiffTCNODE(num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat,n_steps)

        if loop != 0:
            # print(loop)
            model_path = os.path.join(model_dir, f"clu1_{kk}_{loop-1}_3model.h5")
            # torch.save(model.state_dict(), save_path)
            # model_path = "model/alldrugtest/alldata_case1/clu1_" + str(kk) + "_"+str(loop-1)+"_3model.h5"
            model.load_state_dict(torch.load(model_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.SmoothL1Loss()
        model.train()
        for t in range(num_epoch):
            epoch_loss = 0.0
            num_samples = 0
            batch_loss = 0
            # for idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            for idx, (inputs, targets) in enumerate(train_loader):
                # print(inputs.shape,"inputshape")
                # exit()

                loss,control = diffusion_loss_fn(model, inputs,targets, criterion, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,args.batch_size)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                bs = inputs.size(0) if hasattr(inputs, "size") else args.batch_size
                epoch_loss += loss.item() * bs
                num_samples += bs

            avg_loss = epoch_loss / max(1, num_samples)
            # print(f"Epoch [{t+1}/{num_epoch}]  total_loss={epoch_loss:.6f}  avg_loss={avg_loss:.6f}")

            # if t % 400 == 0:
            #     if control == 10:
            #         print(loss,'loss',t,'epoch')
            #     else:
            #         print(control,'loss',t,'epoch')


            # if t == num_epoch-2:
            #     print(loss, 'loss', t, 'epoch')
            #     break
        # model.eval()
        '''
            epoch_loss = 0.0
            num_samples = 0
            batch_loss = 0
            # for idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            for idx, (inputs, targets) in enumerate(train_loader):
                # print(inputs.shape,"inputshape")
                # exit()
                optimizer.zero_grad()
                epoch_loss = 0.0

        # 遍历所有扩散步长
                for t in range(n_steps):
                    if t<num_steps/4:
                        choose = 0
                    else:
                        choose = 1
                    loss, _ = diffusion_loss_fn(model, inputs, targets, criterion,
                                        alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                        t, choose)




                    total_batch_loss += loss

        # 平均化所有步长的 loss
                loss = total_batch_loss / n_steps

                # 反向传播 + 梯度裁剪 + 更新
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # 按 batch size 加权累计 loss
                bs = inputs.size(0) if hasattr(inputs, "size") else args.batch_size
                epoch_loss += loss.item() * bs
                num_samples += bs

            # 按样本数平均
            avg_loss = epoch_loss / max(1, num_samples)

            if t % 400 == 0:
                if control == 10:
                    print(loss,'loss',t,'epoch')
                else:
                    print(control,'loss',t,'epoch')


                # loss,control = diffusion_loss_fn(model, inputs,targets, criterion, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,args.batch_size)





                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

                bs = inputs.size(0) if hasattr(inputs, "size") else args.batch_size
                epoch_loss += loss.item() * bs
                num_samples += bs

            avg_loss = epoch_loss / max(1, num_samples)
            # print(f"Epoch [{t+1}/{num_epoch}]  total_loss={epoch_loss:.6f}  avg_loss={avg_loss:.6f}")

            # if t % 400 == 0:
            #     if control == 10:
            #         print(loss,'loss',t,'epoch')
            #     else:
            #         print(control,'loss',t,'epoch')


            # if t == num_epoch-2:
            #     print(loss, 'loss', t, 'epoch')
            #     break
            '''
        model.eval()

        loss,control = diffusion_loss_fn_test(model, inputs,targets, criterion, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,args.batch_size)
        print(control,'shixuloss','in',loop,'loop')
        # model.state_dict()
        save_path = os.path.join(model_dir, f"clu1_{kk}_{loop}_3model.h5")
        torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(), "model/alldrugtest/alldata_case1/clu1_" + str(kk) + "_"+str(loop)+"_3model.h5")

        # test_dataset
        # exit()

        test_dataset = []
        test_dataset_labels = []

        # for idx, (batch_data, batch_labels,) in enumerate(test_loader):
        #     test_dataset.append(batch_data)
        #     test_dataset_labels.append(batch_labels)
        #
        # test_dataset = torch.cat(test_dataset, dim=0)
        # test_dataset = test_dataset[:,:,:1,:1]



        # test_dataset = torch.from_numpy(all_test.astype(np.float32))
        # test_dataset = test_dataset.squeeze()
        # print(test_dataset.shape)
        # train_loader, valid_loader_non, test_loader_non = generate_dataset(data, args)
        # print(all_test.shape)
        all_test = read_data(all_test)
        test_loader, valid_loader_non, test_loader_non = generate_dataset_test(all_test, args)
        # for idxqq, (inputs, targets) in enumerate(test_loader):

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



        # print('succeed_' + str(kk))
        kk = kk + 1
        # kk += 1



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
    gold = np.load('data/causaltime/causaltime_gen_ver1.0/traffic/graph.npy')
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
    # expression = pd.read_table('data/sim6.csv', sep=',',header=None)
    ####Drug取前三种药物
    expression = np.load('data/causaltime/causaltime_gen_ver1.0/traffic/gen_data.npy')
    print(expression.shape)
    # expression = np.load('data/causaltime/causaltime_gen_ver1.0/medical/graph.npy')
    # print(expression)
    # expression =expression.reshape(480*40, 40)
    expression = expression[:, :, :20]        # shape: (480, 40, 20)
    expression = expression.reshape(-1, 20)  # shape: (480*40, 20)

    # expression =expression.reshape(480*40, 40)
    # expression =expression[1]

    # pd.read_table('data/sim6.csv', sep=',',header=None)
    # print(expression)
    expression =pd.DataFrame(expression)
    print(expression.shape)
    # expression =expression.iloc[40:80,0:20]
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