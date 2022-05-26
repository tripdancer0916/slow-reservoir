import argparse
import copy
import math
import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import yaml

sys.path.append('../')
from dataset.dynamic_state import State
from model import RNN


class BayesianOptimality:
    sigma_sq = 0.5
    phi = np.linspace(-2, 2, 100)
    mu_l_list = np.linspace(-1, 1, 200)

    def __init__(self, config_path, model_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        model_name = os.path.splitext(os.path.basename(config_path))[0]
        print('model_name: ', model_name)

        torch.manual_seed(1)
        self.device = torch.device('cpu')

        self.model = RNN(
            n_in=self.cfg['DATALOADER']['INPUT_NEURON'],
            n_out=1,
            n_hid=self.cfg['MODEL']['SIZE'],
            n_reservoir=self.cfg['MODEL']['RESERVOIR'],
            device=self.device,
            alpha_fast=self.cfg['MODEL']['ALPHA_FAST'],
            alpha_slow=self.cfg['MODEL']['ALPHA_SLOW'],
            sigma_neu=self.cfg['MODEL']['SIGMA_NEU'],
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.sigma_l=math.sqrt(1/1) * 0.5
        # self.g = 1 / (4 * self.sigma_l)
        self.g = 0.1
        # self.sigma_l = 1/8
        
    def make_signal_for_prior(self, time_length, mu_p, sigma_p, uncertainty=0.5):
        input_signals = np.zeros([1, time_length, 100])
    
        state = State(mu=mu_p, sigma=sigma_p)
        signal_input = np.zeros([time_length, 100])
        for t in range(time_length):
            true_signal = state()
            signal_sigma = np.sqrt(1 / self.g) * uncertainty
            signal_mu = np.random.normal(true_signal, signal_sigma)
            
            signal_base = self.g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * self.sigma_sq))
            signal_input[t] = np.random.poisson(signal_base)
        input_signals[0] = signal_input
        
        return input_signals

    def make_sample_signal(self):
        input_signals = np.zeros([200, 1, 100])

        for i, signal_mu in enumerate(self.mu_l_list):
            signal_base = self.g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * self.sigma_sq))
            input_signals[i, 0] = np.random.poisson(signal_base)

        return input_signals

    def bayesian_optimality(self, mu_p, sigma_p, randomize=None):
        input_signal = self.make_signal_for_prior(
            time_length=50, 
            mu_p=mu_p,
            sigma_p=sigma_p,
            uncertainty=0.5,
        )
        inputs = torch.from_numpy(input_signal).float()                                               
        inputs = inputs.to(self.device) 

        hidden_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['SIZE']))
        reservoir_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['RESERVOIR']))

        hidden = torch.from_numpy(hidden_np).float()                               
        hidden = hidden.to(self.device) 
        reservoir = torch.from_numpy(reservoir_np).float()
        reservoir = reservoir.to(self.device)
        hidden_list, _, _, reservoir_list = self.model(inputs, hidden, reservoir, 50) 

        neural_dynamics = hidden_list.cpu().detach().numpy()   
        reservoir_dynamics = reservoir_list.cpu().detach().numpy()

        initial_state = copy.deepcopy(neural_dynamics[0, -1])
        initial_res_state = copy.deepcopy(reservoir_dynamics[0, -1])

        input_signal = self.make_sample_signal()

        if randomize == 'main':
            hidden_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['SIZE']))
            # hidden_np = np.zeros((1, self.cfg['MODEL']['SIZE']))
        else:
            hidden_np = initial_state
        hidden = torch.from_numpy(hidden_np).float()                               
        hidden = hidden.to(self.device)                                                   

        if randomize == 'sub':
            reservoir_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['RESERVOIR']))
            # reservoir_np = np.zeros((1, self.cfg['MODEL']['RESERVOIR']))
        else:
            reservoir_np = initial_res_state
        reservoir = torch.from_numpy(reservoir_np).float()
        reservoir = reservoir.to(self.device)

        inputs = torch.from_numpy(input_signal).float()                                               
        inputs = inputs.to(self.device)                                                                             

        _, output_list, _, _ = self.model(inputs, hidden, reservoir, 1)
        
        bayesian_optimal = mu_p * (self.sigma_l**2/(sigma_p**2+self.sigma_l**2)) + \
            self.mu_l_list * (sigma_p**2/(sigma_p**2+self.sigma_l**2))

        mse = mean_squared_error(output_list[:, 0, 0].detach().numpy(), bayesian_optimal)

        return mse

    def evaluate_optimality(self, randomize=None):
        mu_p_list = np.linspace(-0.5, 0.5, 11)
        sigma_p_list = np.linspace(0, 0.5, 6)
        mse = 0
        count = 0
        for mu_p in mu_p_list:
            for sigma_p in sigma_p_list:
                _mse = self.bayesian_optimality(
                    mu_p=mu_p, 
                    sigma_p=sigma_p,
                    randomize=randomize,
                )
                mse += _mse
                count += 1
        
        # print(f'Mean Squared Error: {mse / count:.5f}')
        
        return mse / count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    print(f'args: {args}')
    bayesian_optimality = BayesianOptimality(
        config_path=args.config_path,
        model_path=args.model_path,
    )
    bayesian_optimality.evaluate_optimality()
