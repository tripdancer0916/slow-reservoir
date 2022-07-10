import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')

from analysis.bayesian_optimality import BayesianOptimality


cfg_dict = {
    1: [
        '20220428_1',
        '20220509_1',
        '20220526_1_2',
    ],
    0.1: [
        '20220428_4',
        '20220509_4',
        '20220526_4_2',
    ],
}

mse_list = np.zeros((2, 6, 3))

tp_list = [0, 0.01, 0.02, 0.04, 0.06, 0.08]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('-sn', '--sample_num', type=int, default=200)
    args = parser.parse_args()
    print(f'args: {args}')
    for i, key in enumerate(cfg_dict):
        for j, tp in enumerate(tp_list):
            for k, cfg_name in enumerate(cfg_dict[key]):
                bayesian_optimality = BayesianOptimality(
                    config_path=f'../cfg/dynamic_state/{cfg_name}.cfg',
                    model_path=f'../trained_model/dynamic_state_random/{cfg_name}/epoch_500.pth',
                    transition_probability=tp,
                )
                mse = bayesian_optimality.evaluate_optimality(sample_num=args.sample_num)
                print(mse)
                mse_list[i, j, k] = mse

    alpha_1_mean = [np.mean(_v_s) for _v_s in mse_list[0]]
    alpha_1_std = [np.std(_v_s) for _v_s in mse_list[0]]

    alpha_01_mean = [np.mean(_v_s) for _v_s in mse_list[1]]
    alpha_01_std = [np.std(_v_s) for _v_s in mse_list[1]]
    
    width = 0.3
    
    plt.figure(constrained_layout=True)
    plt.errorbar(
        tp_list, alpha_1_mean, yerr=alpha_1_std, color='royalblue', 
        linestyle='dashed', capsize=2, fmt='o', label=r'$\alpha_s=1$',
    )
    plt.errorbar(
        tp_list, alpha_01_mean, yerr=alpha_01_std, color='#f7590a', 
        linestyle='solid', capsize=2, fmt='^', label=r'$\alpha_s=0.1$',
    )

    plt.legend(fontsize=16)
    plt.title(f'MSE', fontsize=20)
    # plt.xscale('log')
    # plt.xticks(tp_list)
    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xlabel(r'$p_t$', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.ylim(bottom=0)

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/bayesian_optimality_tp.png', dpi=200)
