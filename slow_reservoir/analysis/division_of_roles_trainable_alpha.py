import argparse
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
import yaml

sys.path.append('../')
from models.simple_rnn import RNNSimpleTrainableAlpha
from analysis.division_of_roles import RoleDivision


class RoleDivisionTrainableAlpha(RoleDivision):
    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if 'ACTIVATE_FUNC' not in cfg['MODEL']:
            cfg['MODEL']['ACTIVATE_FUNC'] = 'relu'
        model_name = os.path.splitext(os.path.basename(config_path))[0]
        print('model_name: ', model_name)
        torch.manual_seed(1)
        device = torch.device('cpu')
        model = RNNSimpleTrainableAlpha(
            n_in=cfg['DATALOADER']['INPUT_NEURON'],
            n_out=1,
            n_hid=cfg['MODEL']['SIZE'],
            device=device,
            sigma_neu=cfg['MODEL']['SIGMA_NEU'],
            activate_func=cfg['MODEL']['ACTIVATE_FUNC'],
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model.eval()
        self.device = device
        self.sigma_l = math.sqrt(1 / 1.25) * 0.5
        self.g = 1.25
        self.cfg = cfg

    def calc_variance(self, sample_num, slow_thres, fast_thres):
        slow_ids = (self.model.alpha.detach().numpy() < slow_thres).nonzero()[0]
        fast_ids = (self.model.alpha.detach().numpy() > fast_thres).nonzero()[0]
        fast_states = np.zeros((sample_num, len(fast_ids)))
        slow_states = np.zeros((sample_num, len(slow_ids)))

        mu_p_list = []
        sigma_p_list = []
        for i in tqdm(range(sample_num)):
            mu_p = np.random.rand() - 0.5
            sigma_p = np.random.rand() * 0.8
            mu_p_list.append(mu_p)
            sigma_p_list.append(sigma_p)
            input_signal = self.make_signal_for_prior(
                time_length=50,
                mu_p=mu_p,
                sigma_p=sigma_p,
                uncertainty=0.5,
                g=1.25,
            )
            inputs = torch.from_numpy(input_signal).float()
            inputs = inputs.to(self.device)
            hidden_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['SIZE']))

            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(self.device)
            hidden_list, _ = self.model(inputs, hidden, 50)

            fast_dynamics = hidden_list.cpu().detach().numpy()[:, :, fast_ids]
            slow_dynamics = hidden_list.cpu().detach().numpy()[:, :, slow_ids]

            fast_states[i] = fast_dynamics[0, -1]
            slow_states[i] = slow_dynamics[0, -1]

        input_signal = self.make_signal_for_prior(
            time_length=50,
            mu_p=0,
            sigma_p=0.4,
            uncertainty=0.5,
            g=1.25,
        )
        inputs = torch.from_numpy(input_signal).float()
        inputs = inputs.to(self.device)
        hidden_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['SIZE']))

        hidden = torch.from_numpy(hidden_np).float()
        hidden = hidden.to(self.device)
        hidden_list, _ = self.model(inputs, hidden, 50)

        input_signal = self.make_sample_signal(0, sample_num)
        inputs = torch.from_numpy(input_signal).float().to(self.device)

        hidden = torch.stack([hidden_list[:, -1] for _ in range(sample_num)], 0)
        hidden = hidden.reshape(500, 80)
        hidden[:, slow_ids] = torch.from_numpy(slow_states).float()
        _, output_list = self.model(inputs, hidden, 1)
        var_slow = np.var(output_list[:, 0, 0].detach().numpy())

        hidden = torch.stack([hidden_list[:, -1] for _ in range(sample_num)], 0)
        hidden = hidden.reshape(500, 80)
        hidden[:, fast_ids] = torch.from_numpy(fast_states).float()
        _, output_list = self.model(inputs, hidden, 1)
        var_fast = np.var(output_list[:, 0, 0].detach().numpy())

        return var_slow, var_fast


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('-sn', '--sample_num', type=int, default=1000)
    parser.add_argument('-st', '--slow_thres', type=float, default=0.2)
    parser.add_argument('-ft', '--fast_thres', type=float, default=0.8)
    args = parser.parse_args()
    print(f'args: {args}')
    role_division = RoleDivisionTrainableAlpha(
        config_path=args.config_path,
        model_path=args.model_path,
    )
    var_slow, var_fast = role_division.calc_variance(
        sample_num=args.sample_num,
        slow_thres=args.slow_thres,
        fast_thres=args.fast_thres,
    )
    print(f'Variation slow: {var_slow:.5f}')
    print(f'Variation fast: {var_fast:.5f}')
