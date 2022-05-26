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
from bayesian_optimal import BayesianOptimality
from model import RNN


parser = argparse.ArgumentParser(description='PyTorch RNN training')
parser.add_argument('config_path', type=str)
parser.add_argument('model_path', type=str)

args = parser.parse_args()
print(f'args: {args}')
bayesian_optimality = BayesianOptimality(
    config_path=args.config_path,
    model_path=args.model_path,
)

mse_normal = bayesian_optimality.evaluate_optimality(randomize=None)
mse_random_main = bayesian_optimality.evaluate_optimality(randomize='main')
mse_random_sub = bayesian_optimality.evaluate_optimality(randomize='sub')

print(f'Mean Squared Error Normal: {mse_normal:.5f}')
print(f'Mean Squared Error Random Main: {mse_random_main:.5f}')
print(f'Mean Squared Error Random Sub: {mse_random_sub:.5f}')