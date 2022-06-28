import argparse
import math
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append('../')
from dataset.dynamic_state_random import State
from models.rnn import RNN, RNNTrainableAlpha


class RoleDivision:
    sigma_sq = 0.5
    phi = np.linspace(-2, 2, 100)
    mu_l_list = np.linspace(-1, 1, 200)

    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if 'ACTIVATE_FUNC' not in cfg['MODEL']:
            cfg['MODEL']['ACTIVATE_FUNC'] = 'relu'
        model_name = os.path.splitext(os.path.basename(config_path))[0]
        print('model_name: ', model_name)
        torch.manual_seed(1)
        device = torch.device('cpu')
        if cfg['MODEL'].get('TRAIN_ALPHA', False):
            model = RNNTrainableAlpha(
                n_in=cfg['DATALOADER']['INPUT_NEURON'],
                n_out=1,
                n_hid=cfg['MODEL']['SIZE'],
                n_reservoir=cfg['MODEL']['RESERVOIR'],
                device=device,
                sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                activate_func=cfg['MODEL']['ACTIVATE_FUNC'],
            ).to(device)
        else:
            model = RNN(
                n_in=cfg['DATALOADER']['INPUT_NEURON'],
                n_out=1,
                n_hid=cfg['MODEL']['SIZE'],
                n_reservoir=cfg['MODEL']['RESERVOIR'],
                device=device,
                alpha_fast=cfg['MODEL']['ALPHA_FAST'],
                alpha_slow=cfg['MODEL']['ALPHA_SLOW'],
                sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                activate_func=cfg['MODEL']['ACTIVATE_FUNC'],
            ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model.eval()
        self.device = device
        self.sigma_l = math.sqrt(1 / 1.25) * 0.5
        self.g = 1.25
        self.cfg = cfg

    def make_signal_for_prior(self, time_length, mu_p, sigma_p, uncertainty=0.5, g=1.25):
        input_signals = np.zeros([1, time_length, 100])

        state = State(mu=mu_p, sigma=sigma_p)

        for i in range(1):
            signal_input = np.zeros([time_length, 100])
            for t in range(time_length):
                true_signal = state()
                signal_sigma = np.sqrt(1 / g) * uncertainty
                signal_mu = np.random.normal(true_signal, signal_sigma)

                signal_base = g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * self.sigma_sq))
                signal_input[t] = np.random.poisson(signal_base)

            input_signals[i] = signal_input

        return input_signals

    def make_sample_signal(self, signal_mu, batch_size, g=1.25):
        input_signals = np.zeros([batch_size, 1, 100])

        signal_base = g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * self.sigma_sq))
        for i in range(batch_size):
            input_signals[i, 0] = np.random.poisson(signal_base)

        return input_signals

    def calc_variance(self, sample_num):
        neural_states = np.zeros((500, 200))
        reservoir_states = np.zeros((500, 50))

        mu_p_list = []
        sigma_p_list = []
        for i in range(sample_num):
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
            reservoir_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['RESERVOIR']))

            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(self.device)
            reservoir = torch.from_numpy(reservoir_np).float()
            reservoir = reservoir.to(self.device)
            hidden_list, _, reservoir_list = self.model(inputs, hidden, reservoir, 50)

            neural_dynamics = hidden_list.cpu().detach().numpy()
            reservoir_dynamics = reservoir_list.cpu().detach().numpy()

            neural_states[i] = neural_dynamics[0, -1]
            reservoir_states[i] = reservoir_dynamics[0, -1]

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
        reservoir_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['RESERVOIR']))

        hidden = torch.from_numpy(hidden_np).float()
        hidden = hidden.to(self.device)
        reservoir = torch.from_numpy(reservoir_np).float()
        reservoir = reservoir.to(self.device)
        hidden_list, _, reservoir_list = self.model(inputs, hidden, reservoir, 50)

        base_neural_state = hidden_list.cpu().detach().numpy()[:, -1]
        base_reservoir_state = reservoir_list.cpu().detach().numpy()[:, -1]

        input_signal = self.make_sample_signal(0, sample_num)
        inputs = torch.from_numpy(input_signal).float().to(self.device)

        hidden = torch.from_numpy(base_neural_state).float().to(self.device)
        reservoir = torch.from_numpy(reservoir_states).float().to(self.device)
        _, output_list, _ = self.model(inputs, hidden, reservoir, 1)

        var_sub = np.var(output_list[:, 0, 0].detach().numpy())

        input_signal = self.make_sample_signal(0, sample_num)
        inputs = torch.from_numpy(input_signal).float().to(self.device)

        hidden = torch.from_numpy(neural_states).float().to(self.device)
        reservoir = torch.from_numpy(base_reservoir_state).float().to(self.device)
        _, output_list, _ = self.model(inputs, hidden, reservoir, 1)

        var_main = np.var(output_list[:, 0, 0].detach().numpy())

        return var_sub, var_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('-sn', '--sample_num', type=int, default=1000)
    args = parser.parse_args()
    print(f'args: {args}')
    role_division = RoleDivision(
        config_path=args.config_path,
        model_path=args.model_path,
    )
    var_sub, var_main = role_division.calc_variance(sample_num=args.sample_num)
    print(f'Variation sub: {var_sub:.5f}')
    print(f'Variation main: {var_main:.5f}')
