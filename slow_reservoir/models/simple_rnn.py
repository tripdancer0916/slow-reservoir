"""Define recurrent neural network"""

import torch
import torch.nn as nn


class RNNSimple(nn.Module):
    def __init__(
        self, n_in, n_out, n_hid,
        device, alpha=1,
        sigma_neu=0.05, activate_func='relu',
    ):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid)
        self.w_hh = nn.Linear(n_hid, n_hid)
        self.w_out = nn.Linear(n_hid, n_out)

        self.sigma_neu = sigma_neu
        self.device = device

        self.alpha = torch.ones(self.n_hid) * alpha
        self.alpha = self.alpha.to(self.device)

        self.activate_func = activate_func

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)

            if self.activate_func == 'relu':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
            elif self.activate_func == 'tanh':
                tmp_hidden = torch.nn.functional.tanh(tmp_hidden)
            elif self.activate_func == 'relu_clamped':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                tmp_hidden = torch.clamp(tmp_hidden, max=2)
            else:
                raise NotImplementedError

            neural_noise = self.make_neural_noise(hidden, self.alpha)
            hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            output = self.w_out(hidden)
            output = torch.clamp(output, min=-2, max=2)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list


class RNNSimpleTrainableAlpha(nn.Module):
    def __init__(
        self, n_in, n_out, n_hid,
        device, sigma_neu=0.05, activate_func='relu',
    ):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid)
        self.w_hh = nn.Linear(n_hid, n_hid)
        self.w_out = nn.Linear(n_hid, n_out)

        self.sigma_neu = sigma_neu
        self.device = device

        self.alpha = nn.Parameter(
            torch.normal(mean=0.5*torch.ones(self.n_hid), std=0.05*torch.ones(self.n_hid)),
        )
        self.alpha = self.alpha.to(self.device)

        self.activate_func = activate_func

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, length):
        num_batch = input_signal.size(0)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(hidden)

            if self.activate_func == 'relu':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
            elif self.activate_func == 'tanh':
                tmp_hidden = torch.nn.functional.tanh(tmp_hidden)
            elif self.activate_func == 'relu_clamped':
                tmp_hidden = torch.nn.functional.relu(tmp_hidden)
                tmp_hidden = torch.clamp(tmp_hidden, max=2)
            else:
                raise NotImplementedError

            neural_noise = self.make_neural_noise(hidden, self.alpha)
            hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            output = self.w_out(hidden)
            output = torch.clamp(output, min=-2, max=2)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list
