"""Generating input and target"""
import random

import numpy as np
import torch.utils.data as data


class State:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)


class DynamicState(data.Dataset):
    def __init__(
                self,
                time_length,
                input_neuron,
                uncertainty,
                state_list,
                transition_probability,
                g_scale=1,
                sigma_sq=5,
                g_min=0.25,
                g_max=1.25,
                batch_size=50,
                ):
        self.time_length = time_length
        self.input_neuron = input_neuron
        self.uncertainty = uncertainty
        self.state_list = state_list
        self.transition_probability = transition_probability
        self.g_scale = g_scale
        self.sigma_sq = sigma_sq
        self.g_min = g_min
        self.g_max = g_max
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size * 5

    def __getitem__(self, item):
        signal_input = np.zeros((self.time_length, self.input_neuron))
        true_signal_list = np.zeros(self.time_length)

        phi = np.linspace(-2, 2, self.input_neuron)
        mu_history = []
        sigma_history = []
        current_state = random.choice(self.state_list)
        mu_history.append(current_state.mu)
        sigma_history.append(current_state.sigma)
        for t in range(self.time_length):
            true_signal = current_state()
            g = np.random.rand() * (self.g_max - self.g_min) + self.g_min
            signal_sigma = np.sqrt(1 / g) * self.uncertainty
            signal_mu = np.random.normal(true_signal, signal_sigma)

            # signal
            signal_base = g * self.g_scale * np.exp(-(signal_mu - phi) ** 2 / (2.0 * (self.sigma_sq**2)))
            signal_input[t] = np.random.poisson(signal_base)
            true_signal_list[t] = true_signal

            if np.random.rand() < self.transition_probability:
                current_state = random.choice([state for state in self.state_list if state != current_state])
                # print(current_state)
            
            mu_history.append(current_state.mu)
            sigma_history.append(current_state.sigma)
        mu_history = np.array(mu_history, dtype='float')
        sigma_history = np.array(sigma_history)

        # print(mu_history)

        return signal_input, true_signal_list, mu_history, sigma_history
