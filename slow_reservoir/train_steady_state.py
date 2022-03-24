"""Training models"""

import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
import yaml

from torch.autograd import Variable

from dataset.steady_state import SteadyState
from model import RNN


def autocorrelation(data, k, device):
    """Returns the autocorrelation of the *k*th lag in a time series data.
    Parameters
    ----------
    data : one dimensional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """

    # Average of y
    y_avg = torch.mean(data, dim=1).to(device)

    # The calculation of numerator
    sum_of_covariance = torch.zeros(y_avg.shape[0]).to(device)
    for i in range(k + 1, data.shape[1]):
        covariance = (data[:, i] - y_avg) * (data[:, i - (k + 1)] - y_avg)
        # print(covariance)
        sum_of_covariance += covariance[:, 0]

    # The calculation of denominator
    sum_of_denominator = torch.zeros(y_avg.shape[0]).to(device)
    for u in range(data.shape[1]):
        denominator = (data[:, u] - y_avg) ** 2
        sum_of_denominator += denominator[:, 0]

    return sum_of_covariance / sum_of_denominator


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    save_path = f'trained_model/steady_state/{model_name}'
    tmp_path = f'trained_model/steady_state/{model_name}/tmp'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)

    # copy config file
    shutil.copyfile(
        config_path, 
        os.path.join(save_path, os.path.basename(config_path)),
    )

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'device: {device}')

    model = RNN(
        n_in=cfg['DATALOADER']['INPUT_NEURON'],
        n_out=1,
        n_hid=cfg['MODEL']['SIZE'],
        n_reservoir=cfg['MODEL']['RESERVOIR'],
        device=device,
        alpha_fast=cfg['MODEL']['ALPHA_FAST'],
        alpha_slow=cfg['MODEL']['ALPHA_SLOW'],
        sigma_neu=cfg['MODEL']['SIGMA_NEU'],
    ).to(device)

    train_dataset = SteadyState(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        pre_mu=cfg['DATALOADER']['PRE_MU'],
        pre_sigma=cfg['DATALOADER']['PRE_SIGMA'],
        g_scale=cfg['DATALOADER']['G_SCALE'],
        sigma_sq=cfg['DATALOADER']['SIGMA_SQ'],
        batch_size=cfg['TRAIN']['BATCHSIZE'],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
        num_workers=2, shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )
    print(model)

    if cfg['TRAIN']['OPT'] == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['TRAIN']['LR'],
            weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
        )
    elif cfg['TRAIN']['OPT'] == 'SGD':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['TRAIN']['LR'],
            weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
            momentum=0.9,
        )
    else:
        ValueError('optimizer must be Adam or SGD')

    a_list = torch.linspace(-2, 2, 100) + 0.02
    a_list = a_list.to(device)
    model.train()
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        for i, data in enumerate(train_dataloader):
            inputs, true_signal, signal_mu, signal_sigma = data
            inputs =  Variable(inputs.float()).to(device)
            true_signal = Variable(true_signal.float()).to(device)

            hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(device)

            reservoir_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['RESERVOIR']))
            reservoir = torch.from_numpy(reservoir_np).float()
            reservoir = reservoir.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            reservoir = reservoir.detach()

            _, output_list, prior_list = model(
                inputs, 
                hidden, 
                reservoir,
                cfg['DATALOADER']['TIME_LENGTH'],
            )

            # Generate target
            # Likelihood
            likelihood = np.zeros((cfg['TRAIN']['BATCHSIZE'], cfg['DATALOADER']['INPUT_NEURON']))
            phi = np.linspace(-2, 2, cfg['DATALOADER']['INPUT_NEURON'])
            for i in range(cfg['TRAIN']['BATCHSIZE']):
                likelihood[i] = np.exp(-(signal_mu[i].item() - phi) ** 2 / (2.0 * (signal_sigma[i].item()**2)))
                # if i == 0:
                #     print(signal_mu[i].item())
                #     print(likelihood[i])

            # Posterior and Maximum a posteriori
            posterior = np.zeros((cfg['TRAIN']['BATCHSIZE'], 90, cfg['DATALOADER']['INPUT_NEURON']))
            map = np.zeros((cfg['TRAIN']['BATCHSIZE'], 90))
            for i in range(cfg['TRAIN']['BATCHSIZE']):
                for t in range(20, 110):
                    # もしかしたらここまで自動微分を通した方がいいかもしれない...
                    # posterior[i, t-20] = likelihood[i] * prior_list.detach().cpu().numpy()[i, t]
                    map[i, t-20] = np.argmax(likelihood[i] * prior_list.detach().cpu().numpy()[i, t]) / 25 - 2
                    # if i == 0 and t <= 30:
                    #     print('prior: ', np.argmax(prior_list.detach().cpu().numpy()[0, t]) / 25 - 2)
                    #     print('likelihood: ', np.argmax(likelihood[i]) / 25 - 2)
            
            # print(output_list[:, 20:].shape)
            # print(torch.from_numpy(map).to(device).shape)
            map_loss = torch.nn.MSELoss()(
                output_list[:, 20:, 0], 
                torch.from_numpy(map).float().to(device),
            )
            true_value_loss = torch.nn.MSELoss()(
                torch.mean(output_list[:, 20:, 0], dim=1),
                true_signal,
            )

            true_value_loss_coef = cfg['TRAIN']['TRUE_VALUE_LOSS_COEF']
            loss = map_loss + true_value_loss * true_value_loss_coef
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(tmp_path, f'epoch_{epoch}.pth'))
        if epoch - 10 > 0 and epoch - 10 % cfg['TRAIN']['NUM_SAVE_EPOCH'] != 0:
            os.remove(os.path.join(tmp_path, f'epoch_{epoch-10}.pth'))

        if epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            print(f'true_signal: {true_signal[0].item():.3f}')
            print(f'signal_mu:  {signal_mu[0].item():.3f}')
            print('output: ', output_list[0, -10:, 0].detach().cpu().numpy())
            print('map: ', map[0, -10:])
            print(
                f'Train Epoch: {epoch}, MapLoss: {map_loss.item():.3f}, '
                f'TrueValueLoss: {true_value_loss.item():.3f}',
            )
            if math.isnan(loss.item()):
                print('Rewinding due to nan.')
                model = RNN(
                    n_in=cfg['DATALOADER']['INPUT_NEURON'],
                    n_out=1,
                    n_hid=cfg['MODEL']['SIZE'],
                    n_reservoir=cfg['MODEL']['RESERVOIR'],
                    device=device,
                    alpha_fast=cfg['MODEL']['ALPHA_FAST'],
                    alpha_slow=cfg['MODEL']['ALPHA_SLOW'],
                    sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                ).to(device)
                model_path = os.path.join(tmp_path, f'epoch_{epoch-5}.pth')
                model.load_state_dict(torch.load(model_path, map_location=device))
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
