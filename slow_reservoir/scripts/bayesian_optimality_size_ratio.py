import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

sys.path.append('../')

from analysis.bayesian_optimality import BayesianOptimality


cfg_dict = {
    '10/240': [
        '20230905_pt-0.03_sub-10_1',
        '20230905_pt-0.03_sub-10_2',
        '20230905_pt-0.03_sub-10_3',
    ],
    '50/200': [
        '20220428_4',
        '20220509_4',
        '20220526_4_2',
    ],
    '100/150': [
        '20230905_pt-0.03_sub-100_1',
        '20230905_pt-0.03_sub-100_2',
        '20230905_pt-0.03_sub-100_3',
    ],
    '150/100': [
        '20230905_pt-0.03_sub-150_1',
        '20230905_pt-0.03_sub-150_2',
        '20230905_pt-0.03_sub-150_3',
    ],
    '200/50': [
        '20230905_pt-0.03_sub-200_1',
        '20230905_pt-0.03_sub-200_2',
        '20230905_pt-0.03_sub-200_3',
    ],
    '240/10': [
        '20230905_pt-0.03_sub-240_1',
        '20230905_pt-0.03_sub-240_2',
        '20230905_pt-0.03_sub-240_3',
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
    labels = ['10/240', '50/200', '100/150', '150/100', '200/50', '240/10']
    
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
    plt.savefig(f'results/bayesian_optimality_p_t={args.transition_probability}_ms_ratio.png', dpi=300)
