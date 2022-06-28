"""Generating input and target"""
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
        transition_probability,
        sigma_sq=5,
        g_min=0.25,
        g_max=1.25,
        batch_size=50,
    ):
        self.time_length = time_length
        self.input_neuron = input_neuron
        self.uncertainty = uncertainty
        self.transition_probability = transition_probability
        self.sigma_sq = sigma_sq
        self.g_min = g_min
        self.g_max = g_max
        self.batch_size = batch_size

        self.phi = np.linspace(-2, 2, self.input_neuron)

    def __len__(self):
        return self.batch_size * 5

    def __getitem__(self, item):
        signal_input = np.zeros((self.time_length, self.input_neuron))
        true_signal_list = np.zeros(self.time_length)
        mu_history = []
        sigma_history = []
        current_state = State(mu=np.random.rand()-0.5, sigma=np.random.rand()*0.8)
        mu_history.append(current_state.mu)
        sigma_history.append(current_state.sigma)
        g = np.random.rand() * (self.g_max - self.g_min) + self.g_min
        signal_sigma = np.sqrt(1 / g) * self.uncertainty
        for t in range(self.time_length):
            true_signal = current_state()
            signal_mu = np.random.normal(true_signal, signal_sigma)

            # signal
            signal_base = g * np.exp(-(signal_mu - self.phi) ** 2 / (2.0 * (self.sigma_sq**2)))
            signal_input[t] = np.random.poisson(signal_base)
            true_signal_list[t] = true_signal

            if np.random.rand() < self.transition_probability:
                current_state = State(mu=np.random.rand()-0.5, sigma=np.random.rand()*0.8)

            mu_history.append(current_state.mu)
            sigma_history.append(current_state.sigma)
        mu_history = np.array(mu_history, dtype='float')
        sigma_history = np.array(sigma_history)

        return signal_input, true_signal_list, mu_history, sigma_history
