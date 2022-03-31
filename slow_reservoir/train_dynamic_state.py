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

from dataset.dynamic_state import DynamicState, State
from model import RNN


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    save_path = f'trained_model/dynamic_state/{model_name}'
    tmp_path = f'{save_path}/tmp'
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
    # print(f'device: {device}')

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

    state_list = [
        State(mu=0, sigma=0.2),
        State(mu=0.5, sigma=0.4),
        State(mu=-0.5, sigma=0.1),
    ]
    train_dataset = DynamicState(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        state_list=state_list,
        transition_probability=0.03,
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
    phi = np.linspace(-2, 2, cfg['DATALOADER']['INPUT_NEURON'])
    model.train()
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        for i, data in enumerate(train_dataloader):
            inputs, true_signal_list, mu_history, sigma_history = data
            # print(mu_history.shape)
            inputs =  Variable(inputs.float()).to(device)
            true_signal_list = Variable(true_signal_list.float()).to(device)

            # print(true_signal_list.shape)

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
            prior_loss = 0
            true_value_loss = 0
            for i in range(cfg['TRAIN']['BATCHSIZE']):
                for t in range(20, 120):
                    target = np.exp(-(mu_history[i, t].item() - phi) ** 2 / (2.0 * (sigma_history[i, t].item()**2)))
                    target /= np.sum(target)

                    prior_loss += torch.nn.MSELoss()(
                        prior_list[i, t],
                        torch.from_numpy(target).float().to(device),
                    )

                    true_value_loss += torch.nn.MSELoss()(
                        output_list[i, t, 0],
                        true_signal_list[i, t],
                    )

            true_value_loss_coef = cfg['TRAIN']['TRUE_VALUE_LOSS_COEF']
            loss = prior_loss + true_value_loss * true_value_loss_coef
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(tmp_path, f'epoch_{epoch}.pth'))
        if epoch - 10 > 0 and epoch - 10 % cfg['TRAIN']['NUM_SAVE_EPOCH'] != 0:
            os.remove(os.path.join(tmp_path, f'epoch_{epoch-10}.pth'))

        if epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            print(f'true_signal: ', true_signal_list[0, -10:].detach().cpu().numpy())
            # print(f'signal_mu:  {signal_mu[0].item():.3f}')
            print('output: ', output_list[0, -10:, 0].detach().cpu().numpy())
            # print('map: ', map[0, -10:])
            # print('prior_list: ', prior_list[0, -10:, 65:75].detach().cpu().numpy())
            print(
                f'Train Epoch: {epoch}, PriorLoss: {prior_loss.item():.3f}, '
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
