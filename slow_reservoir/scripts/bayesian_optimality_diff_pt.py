import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

from analysis.bayesian_optimality import BayesianOptimality


cfg_dict = {
    1: [
        '20220713_pt_03_a_1',
        '20220713_pt_03_a_2',
        '20220713_pt_03_a_3',
    ],
    0.5: [
        '20220713_pt_03_b_1',
        '20220713_pt_03_b_2',
        '20220713_pt_03_b_3',
    ],
    0.4: [
        '20220713_pt_03_c_1',
        '20220713_pt_03_c_2',
        '20220713_pt_03_c_3',
    ],
    0.3: [
        '20220713_pt_03_d_1',
        '20220713_pt_03_d_2',
        '20220713_pt_03_d_3',
    ],
    0.2: [
        '20220713_pt_03_e_1',
        '20220713_pt_03_e_2',
        '20220713_pt_03_e_3',
    ],
    0.1: [
        '20220713_pt_03_f_1',
        '20220713_pt_03_f_2',
        '20220713_pt_03_f_3',
    ],
}


mse_list = np.zeros((6, 3))
mse_list_simple = np.zeros((6, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('-tp', '--transition_probability', type=float, default=0.3)
    parser.add_argument('-sn', '--sample_num', type=int, default=200)
    args = parser.parse_args()
    print(f'args: {args}')
    for i, key in enumerate(cfg_dict):
        for j, cfg_name in enumerate(cfg_dict[key]):
            bayesian_optimality = BayesianOptimality(
                config_path=f'../cfg/dynamic_state/{cfg_name}.cfg',
                model_path=f'../trained_model/dynamic_state_random/{cfg_name}/epoch_500.pth',
                transition_probability=args.transition_probability,
            )
            mse = bayesian_optimality.evaluate_optimality(sample_num=args.sample_num)
            print(mse)
            mse_list[i, j] = mse

    mse_mean = [np.mean(_v_s) for _v_s in mse_list]
    mse_std = [np.std(_v_s) for _v_s in mse_list] 
    left = np.arange(len(mse_mean))

    labels = [1, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    width = 0.3
    
    plt.figure(constrained_layout=True)
    plt.errorbar(left, mse_mean, yerr=mse_std, color='b', capsize=2, linestyle='None', fmt='o', label='RNN with modular structure')
    
    plt.title(r'$MSE(y, y_{opt})$', fontsize=28)

    plt.xticks(left, labels)
    plt.xlabel(r'$\alpha_s$', fontsize=24)
    plt.ylabel('MSE', fontsize=24)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.ylim([0, 0.065])
    plt.legend(fontsize=20)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/bayesian_optimality_train_p_t=0.3_test_p_t={args.transition_probability}.png', dpi=300)
