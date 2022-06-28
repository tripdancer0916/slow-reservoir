"""Define recurrent neural network"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
                self, n_in, n_out, n_hid, n_reservoir, 
                device, alpha_fast=1, alpha_slow=0.1, jij_std=0.045,
                sigma_neu=0.05, activate_func='relu',
                ):
        super(RNN, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_reservoir = n_reservoir
        self.w_in = nn.Linear(n_in, n_hid)
        self.w_hh = nn.Linear(n_hid, n_hid)
        self.w_out = nn.Linear(n_hid, n_out)

        self.w_fs = nn.Linear(n_hid, n_reservoir)
        self.w_sf = nn.Linear(n_reservoir, n_hid)
        self.w_reservoir = nn.Linear(n_reservoir, n_reservoir)
        self.w_prior = nn.Linear(n_reservoir, n_in)
        
        self.sigma_neu = sigma_neu
        self.jij_std = jij_std
        self.device = device

        self.alpha_fast = torch.ones(self.n_hid) * alpha_fast
        self.alpha_fast = self.alpha_fast.to(self.device)

        self.alpha_slow = torch.ones(self.n_reservoir) * alpha_slow
        self.alpha_slow = self.alpha_slow.to(self.device)

        self.activate_func = activate_func

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, reservoir, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        reservoir_list = torch.zeros(length, num_batch, self.n_reservoir).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden) + self.w_sf(reservoir)
            tmp_reservoir = self.w_fs(hidden) + self.w_reservoir(reservoir)

            if self.activate_func == 'relu':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                tmp_reservoir = torch.nn.functional.relu(tmp_reservoir)
            elif self.activate_func == 'tanh':
                tmp_hidden = torch.nn.functional.tanh(tmp_hidden) 
                tmp_reservoir = torch.nn.functional.tanh(tmp_reservoir)
            elif self.activate_func == 'relu_clamped':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                tmp_hidden = torch.clamp(tmp_hidden, max=2)
                tmp_reservoir = torch.nn.functional.relu(tmp_reservoir)
                tmp_reservoir = torch.clamp(tmp_reservoir, max=2)
            else:
                raise NotImplementedError

            neural_noise = self.make_neural_noise(hidden, self.alpha_fast)
            reservoir_noise = self.make_neural_noise(reservoir, self.alpha_slow)
            hidden = (1 - self.alpha_fast) * hidden + self.alpha_fast * tmp_hidden + neural_noise
            reservoir = (1 - self.alpha_slow) * reservoir + self.alpha_slow * tmp_reservoir + reservoir_noise
            output = self.w_out(hidden)
            output = torch.clamp(output, min=-2, max=2)
            hidden_list[t] = hidden
            reservoir_list[t] = reservoir
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        reservoir_list = reservoir_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, reservoir_list


class RNNTrainableAlpha(nn.Module):
    def __init__(
                self, n_in, n_out, n_hid, n_reservoir, 
                device, sigma_neu=0.05, activate_func='relu',
                ):
        super(RNNTrainableAlpha, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_reservoir = n_reservoir
        self.w_in = nn.Linear(n_in, n_hid)
        self.w_hh = nn.Linear(n_hid, n_hid)
        self.w_out = nn.Linear(n_hid, n_out)
        self.w_fs = nn.Linear(n_hid, n_reservoir)
        self.w_sf = nn.Linear(n_reservoir, n_hid)
        self.w_reservoir = nn.Linear(n_reservoir, n_reservoir)
        self.w_prior = nn.Linear(n_reservoir, n_in)
        
        self.sigma_neu = sigma_neu
        self.device = device

        self.alpha_fast = nn.Parameter(torch.rand(self.n_hid))
        self.alpha_slow = nn.Parameter(torch.rand(self.n_reservoir))

        self.activate_func = activate_func

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, reservoir, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        reservoir_list = torch.zeros(length, num_batch, self.n_reservoir).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden) + self.w_sf(reservoir)
            tmp_reservoir = self.w_fs(hidden) + self.w_reservoir(reservoir)

            if self.activate_func == 'relu':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                tmp_reservoir = torch.nn.functional.relu(tmp_reservoir)
            elif self.activate_func == 'tanh':
                tmp_hidden = torch.nn.functional.tanh(tmp_hidden) 
                tmp_reservoir = torch.nn.functional.tanh(tmp_reservoir)
            elif self.activate_func == 'relu_clamped':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                tmp_hidden = torch.clamp(tmp_hidden, max=2)
                tmp_reservoir = torch.nn.functional.relu(tmp_reservoir)
                tmp_reservoir = torch.clamp(tmp_reservoir, max=2)
            else:
                raise NotImplementedError

            neural_noise = self.make_neural_noise(hidden, self.alpha_fast)
            reservoir_noise = self.make_neural_noise(reservoir, self.alpha_slow)
            hidden = (1 - self.alpha_fast) * hidden + self.alpha_fast * tmp_hidden + neural_noise
            reservoir = (1 - self.alpha_slow) * reservoir + self.alpha_slow * tmp_reservoir + reservoir_noise
            output = self.w_out(hidden)
            output = torch.clamp(output, min=-2, max=2)
            hidden_list[t] = hidden
            reservoir_list[t] = reservoir
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        reservoir_list = reservoir_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, reservoir_list
