"""Generating input and target"""

import numpy as np
import torch.utils.data as data


class SteadyState(data.Dataset):
    def __init__(
                self,
                time_length,
                input_neuron,
                uncertainty,
                pre_mu,
                pre_sigma,
                g_scale=1,
                sigma_sq=5,
                g_min=0.25,
                g_max=1.25,
                batch_size=50,
                ):
        self.time_length = time_length
        self.input_neuron = input_neuron
        self.uncertainty = uncertainty
        self.pre_mu = pre_mu
        self.pre_sigma = pre_sigma
        self.g_scale = g_scale
        self.sigma_sq = sigma_sq
        self.g_min = g_min
        self.g_max = g_max
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size * 5

    def __getitem__(self, item):
        signal_input = np.zeros((self.time_length, self.input_neuron))

        phi = np.linspace(-2, 2, self.input_neuron)

        true_signal = np.random.normal(self.pre_mu, self.pre_sigma)
        g = np.random.rand() * (self.g_max - self.g_min) + self.g_min
        signal_sigma = np.sqrt(1 / g) * self.uncertainty
        signal_mu = np.random.normal(true_signal, signal_sigma)

        # signal
        signal_base = g * self.g_scale * np.exp(-(signal_mu - phi) ** 2 / (2.0 * self.sigma_sq))
        for t in range(self.time_length):
            signal_input[t] = np.random.poisson(signal_base)

        signal_mu = np.expand_dims(signal_mu, axis=0)
        signal_sigma = np.expand_dims(signal_sigma, axis=0)

        return signal_input, true_signal, signal_mu, signal_sigma
