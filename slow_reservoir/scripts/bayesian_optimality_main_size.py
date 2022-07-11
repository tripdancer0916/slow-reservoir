import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

from analysis.bayesian_optimality import BayesianOptimality


cfg_dict = {
    25: [
        '20220707_4_main_25_1',
        '20220707_4_main_25_2',
        '20220707_4_main_25_3',
    ],
    50: [
        '20220707_4_main_50_1',
        '20220707_4_main_50_2',
        '20220707_4_main_50_3',
    ],
    100: [
        '20220707_4_main_100_1',
        '20220707_4_main_100_2',
        '20220707_4_main_100_3',
    ],
    200: [
        '20220428_4',
        '20220509_4',
        '20220526_4_2',
    ],
    400: [
        '20220707_4_main_400_1',
        '20220707_4_main_400_2',
        '20220707_4_main_400_3',
    ],
}

mse_list = np.zeros((5, 3))


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
    labels = [25, 50, 100, 200, 400]
    
    width = 0.3
    
    plt.figure(constrained_layout=True)
    plt.errorbar(left, mse_mean, yerr=mse_std, color='b', capsize=2, linestyle='None', fmt='o')
    plt.title(f'MSE p_t={args.transition_probability}', fontsize=20)

    plt.xticks(left, labels)
    plt.xlabel('Size of Main-module', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/bayesian_optimality_mainsize_p_t={args.transition_probability}.png', dpi=200)
