"""Define recurrent neural network"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
                self, n_in, n_out, n_hid, n_reservoir, 
                device, alpha_fast=1, alpha_slow=0.1, jij_std=0.045,
                sigma_neu=0.05,
                ):
        super(RNN, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_reservoir = n_reservoir
        self.w_in = nn.Linear(n_in, n_hid)
        self.w_hh = nn.Linear(n_hid, n_hid)
        self.w_out = nn.Linear(n_hid, n_out)

        # Fixed weights
        self.w_fs = nn.Linear(n_hid, n_reservoir)
        self.w_sf = nn.Linear(n_reservoir, n_hid)
        self.w_reservoir = nn.Linear(n_reservoir, n_reservoir)
        self.w_prior = nn.Linear(n_reservoir, n_in)
        # self.w_prior = nn.Linear(n_reservoir, 1)
        
        self.sigma_neu = sigma_neu
        self.jij_std = jij_std
        self.device = device

        self.alpha_fast = torch.ones(self.n_hid) * alpha_fast
        self.alpha_fast = self.alpha_fast.to(self.device)

        self.alpha_slow = torch.ones(self.n_reservoir) * alpha_slow
        self.alpha_slow = self.alpha_slow.to(self.device)

        # self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.w_hh.weight, -self.jij_std, self.jij_std)

        # fixed_weight_list = [self.w_fs, self.w_sf, self.w_reservoir, self.w_prior]
        fixed_weight_list = [self.w_prior]
        for weight in fixed_weight_list:
            for p in weight.parameters():
                p.required_grad = False

    def change_alpha_fast(self, new_alpha_fast):
        self.alpha_fast = torch.ones(self.n_hid) * new_alpha_fast
        self.alpha_fast = self.alpha_fast.to(self.device)

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, reservoir, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        reservoir_list = torch.zeros(length, num_batch, self.n_reservoir).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)
        prior_list = torch.zeros(length, num_batch, self.n_in).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden) + self.w_sf(reservoir)
            tmp_hidden = torch.nn.functional.relu(tmp_hidden)

            tmp_reservoir = self.w_fs(hidden) + self.w_reservoir(reservoir)
            tmp_reservoir = torch.nn.functional.relu(tmp_reservoir)

            neural_noise = self.make_neural_noise(hidden, self.alpha_fast)
            hidden = (1 - self.alpha_fast) * hidden + self.alpha_fast * tmp_hidden + neural_noise
            reservoir = (1 - self.alpha_slow) * reservoir + self.alpha_slow * tmp_reservoir
            output = self.w_out(hidden)
            # pred_prior = torch.nn.Softmax(dim=1)(self.w_prior(reservoir))
            pred_prior = torch.nn.Sigmoid()(self.w_prior(reservoir))
            # pred_prior = self.w_prior(reservoir)
            output = torch.clamp(output, min=-2, max=2)
            hidden_list[t] = hidden
            reservoir_list[t] = reservoir
            output_list[t] = output
            prior_list[t] = pred_prior

        hidden_list = hidden_list.permute(1, 0, 2)
        reservoir_list = reservoir_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)
        prior_list = prior_list.permute(1, 0, 2)

        return hidden_list, output_list, prior_list


class RNNSimple(nn.Module):
    def __init__(
                self, n_in, n_out, n_hid, 
                device, alpha=1, jij_std=0.045,
                sigma_neu=0.05,
                ):
        super(RNNSimple, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid)
        self.w_hh = nn.Linear(n_hid, n_hid)
        self.w_out = nn.Linear(n_hid, n_out)
        
        self.sigma_neu = sigma_neu
        self.jij_std = jij_std
        self.device = device

        self.alpha = torch.ones(self.n_hid) * alpha
        self.alpha = self.alpha.to(self.device)

    def initialize_weights(self):
        nn.init.uniform_(self.w_hh.weight, -self.jij_std, self.jij_std)

    def change_alpha(self, new_alpha):
        self.alpha = torch.ones(self.n_hid) * new_alpha
        self.alpha = self.alpha.to(self.device)

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)
            tmp_hidden = torch.nn.functional.relu(tmp_hidden)
            neural_noise = self.make_neural_noise(hidden, self.alpha)
            hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            output = self.w_out(hidden)
            output = torch.clamp(output, min=-2, max=2)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden
