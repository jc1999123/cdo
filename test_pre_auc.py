import pandas as pd

from do_result import pre_A_1

import numpy as np

gold = np.loadtxt('data/sim6_gold.txt', dtype='str')
# A_next = pre_A_1(f1_result, f2_result, gold)
kfold =3
kk =3
loop =0
i = 0
f1_result = np.zeros((10,10, k_fold))
f2_result = np.zeros((10,10, k_fold))
# f1 = pd.read_csv(
#     'model/sim6/clu1' + str(kk) + str(i) + "_" + str(loop) + 'mse0.001_adam.csv')
# print(f1)
# exit()
for i in range(kfold):
    f1_result[:,:,i]
    a =pd.read_csv('model/sim6_2/tcntest1/clu1'+ str(kk)  + str(i) + "_"+str(loop)+'mse0.001_adam.csv')
    b =pd.read_csv('model/sim6_2/tcntest1/clu1'+ str(kk)  + str(i) + "_"+str(loop)+'mse0.001_adam.csv')
