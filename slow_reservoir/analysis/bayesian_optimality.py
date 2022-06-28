import argparse
import copy
import math
import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm
import yaml

sys.path.append('../')
from dataset.dynamic_state_random import State
from models.rnn import RNN, RNNTrainableAlpha
from models.simple_rnn import RNNSimple, RNNSimpleTrainableAlpha


class BayesianOptimality:
    sigma_sq = 0.5
    phi = np.linspace(-2, 2, 100)
    mu_l_list = np.linspace(-1, 1, 200)

    def __init__(self, config_path, model_path, transition_probability):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        if 'ACTIVATE_FUNC' not in cfg['MODEL']:
            cfg['MODEL']['ACTIVATE_FUNC'] = 'relu'

        model_name = os.path.splitext(os.path.basename(config_path))[0]
        print('model_name: ', model_name)

        torch.manual_seed(1)
        device = torch.device('cpu')

        if cfg['MODEL'].get('TRAIN_ALPHA', False):
            if cfg['MODEL']['RESERVOIR'] == 0:
                model = RNNSimpleTrainableAlpha(
                    n_in=cfg['DATALOADER']['INPUT_NEURON'],
                    n_out=1,
                    n_hid=cfg['MODEL']['SIZE'],
                    device=device,
                    sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                    activate_func=cfg['MODEL']['ACTIVATE_FUNC'],
                ).to(device)
            else:
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
            if cfg['MODEL']['RESERVOIR'] == 0:
                model = RNNSimple(
                    n_in=cfg['DATALOADER']['INPUT_NEURON'],
                    n_out=1,
                    n_hid=cfg['MODEL']['SIZE'],
                    device=device,
                    alpha=cfg['MODEL']['ALPHA_FAST'],
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

        self.sigma_l=math.sqrt(1/1.25) * 0.5
        self.g = 1.25

        self.transition_probability = transition_probability
        self.cfg = cfg
        
    def make_signal_for_prior(self, time_length, uncertainty=0.5):
        input_signals = np.zeros([1, time_length, 100])
    
        mu_p = np.random.rand() - 0.5
        sigma_p = np.random.rand()*0.8
        current_state = State(mu=mu_p, sigma=sigma_p)
        signal_input = np.zeros([time_length, 100])
        for t in range(time_length):
            true_signal = current_state()
            signal_sigma = np.sqrt(1 / self.g) * uncertainty
            signal_mu = np.random.normal(true_signal, signal_sigma)
            
            signal_base = self.g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * self.sigma_sq))
            signal_input[t] = np.random.poisson(signal_base)

            if np.random.rand() < self.transition_probability:
                mu_p = np.random.rand() - 0.5
                sigma_p = np.random.rand()*0.8
                current_state = State(mu=mu_p, sigma=sigma_p)

        input_signals[0] = signal_input
        
        return input_signals, mu_p, sigma_p

    def make_sample_signal(self):
        input_signals = np.zeros([200, 1, 100])

        for i, signal_mu in enumerate(self.mu_l_list):
            signal_base = self.g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * self.sigma_sq))
            input_signals[i, 0] = np.random.poisson(signal_base)

        return input_signals

    def bayesian_optimality(self, randomize=None):
        input_signal, mu_p, sigma_p = self.make_signal_for_prior(
            time_length=120, 
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

        if self.cfg['MODEL']['RESERVOIR'] == 0:
            hidden_list, _ = self.model(inputs, hidden, reservoir, 120)
        else:  
            hidden_list, _, reservoir_list = self.model(inputs, hidden, reservoir, 120) 
            reservoir_dynamics = reservoir_list.cpu().detach().numpy()
            initial_res_state = copy.deepcopy(reservoir_dynamics[0, -1])

            if randomize == 'sub':
                reservoir_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['RESERVOIR']))
            else:
                reservoir_np = initial_res_state
            reservoir = torch.from_numpy(reservoir_np).float()
            reservoir = reservoir.to(self.device)

        neural_dynamics = hidden_list.cpu().detach().numpy()   
        initial_state = copy.deepcopy(neural_dynamics[0, -1])
        
        input_signal = self.make_sample_signal()

        if randomize == 'main':
            hidden_np = np.random.normal(0, 0.5, size=(1, self.cfg['MODEL']['SIZE']))
        else:
            hidden_np = initial_state
        hidden = torch.from_numpy(hidden_np).float()                               
        hidden = hidden.to(self.device)                                                   

        inputs = torch.from_numpy(input_signal).float()                                               
        inputs = inputs.to(self.device)                                                                             

        if self.cfg['MODEL']['RESERVOIR'] == 0:
            _, output_list = self.model(inputs, hidden, reservoir, 1)
        else:
            _, output_list, _ = self.model(inputs, hidden, reservoir, 1)
        
        bayesian_optimal = mu_p * (self.sigma_l**2/(sigma_p**2+self.sigma_l**2)) + \
            self.mu_l_list * (sigma_p**2/(sigma_p**2+self.sigma_l**2))

        mse = mean_squared_error(output_list[:, 0, 0].detach().numpy(), bayesian_optimal)

        return mse

    def evaluate_optimality(self, randomize=None, sample_num=1000):
        mse = 0
        for _ in tqdm(range(sample_num)):
            _mse = self.bayesian_optimality(randomize=randomize)
            mse += _mse

        return mse / sample_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('-tp', '--transition_probability', type=float, default=0.03)
    parser.add_argument('-sn', '--sample_num', type=int, default=1000)
    args = parser.parse_args()
    print(f'args: {args}')
    bayesian_optimality = BayesianOptimality(
        config_path=args.config_path,
        model_path=args.model_path,
        transition_probability=args.transition_probability,
    )
    mse = bayesian_optimality.evaluate_optimality(sample_num=args.sample_num)
    print(f'Mean Squared Error: {mse:.5f}')
