import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

from analysis.division_of_roles import RoleDivision


cfg_dict = {
    1: [
        '20220428_1',
        '20220509_1',
        '20220526_1_2',
    ],
    0.5: [
        '20220428_2',
        '20220509_2',
        '20220526_2_2',
    ],
    0.2: [
        '20220428_3',
        '20220509_3',
        '20220526_3_2',
    ],
    0.1: [
        '20220428_4',
        '20220509_4',
        '20220526_4_2',
    ],
    0.05: [
        '20220428_5',
        '20220509_5',
        '20220526_5_2',
    ],
    0.01: [
        '20220428_6',
        '20220509_6',
        '20220526_6_2',
    ],
}

v_s_list = np.zeros((6, 3))
v_m_list = np.zeros((6, 3))

for i, key in enumerate(cfg_dict):
    for j, cfg_name in enumerate(cfg_dict[key]):
        role_division = RoleDivision(
            config_path=f'../cfg/dynamic_state/{cfg_name}',
            model_path=f'../trained_model/dynamic_state_random/{cfg_name}/epoch_500.pth',
        )
        var_sub, var_main = role_division.calc_variance(sample_num=500)
        v_s_list[i, j] = var_sub
        v_m_list[i, j] = var_main

var_s_mean = [np.mean(_v_s) for _v_s in v_s_list]  
var_m_mean = [np.mean(_v_m) for _v_m in v_m_list]  

var_s_std = [np.std(_v_s) for _v_s in v_s_list]  
var_m_std = [np.std(_v_m) for _v_m in v_m_list]  

left = np.arange(len(var_s_mean))  # numpyで横軸を設定
labels = [1, 0.5, 0.2, 0.1, 0.05, 0.01]
 
width = 0.3
 
plt.bar(left, var_s_mean, yerr=var_s_std, color='b', width=width, capsize = 2, align='center')
plt.bar(left+width, var_m_mean, yerr=var_m_std, color='r', width=width, capsize = 2, align='center')
 
plt.xticks(left + width/2, labels)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

os.makedirs('results', exist_ok=True)
plt.savefig('division_of_roles.png', dpi=200)
