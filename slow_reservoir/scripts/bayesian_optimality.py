import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

from analysis.bayesian_optimality import BayesianOptimality


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

mse_list = np.zeros((6, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('-tp', '--transition_probability', type=float, default=0.03)
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
    labels = [1, 0.5, 0.2, 0.1, 0.05, 0.01]
    
    width = 0.3
    
    plt.figure(constrained_layout=True)
    plt.errorbar(left, mse_mean, yerr=mse_std, color='b', capsize=2, linestyle='None', fmt='o')
    plt.title(f'MSE p_t={args.transition_probability}', fontsize=20)

    plt.xticks(left, labels)
    plt.xlabel(r'$\alpha_s$', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/bayesian_optimality_p_t={args.transition_probability}.png', dpi=200)
