"""Training models"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
import yaml
from dataset.dynamic_state_random import DynamicState
from models.rnn import RNN, RNNTrainableAlpha
from models.simple_rnn import RNNSimple, RNNSimpleTrainableAlpha
from torch.autograd import Variable


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'ACTIVATE_FUNC' not in cfg['MODEL']:
        cfg['MODEL']['ACTIVATE_FUNC'] = 'relu'
    if 'TRANSITION_PROB' not in cfg['DATALOADER']:
        cfg['DATALOADER']['TRANSITION_PROB'] = 0.03

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    save_path = f'trained_model/dynamic_state_random/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(
        config_path,
        os.path.join(save_path, os.path.basename(config_path)),
    )

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')

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

    train_dataset = DynamicState(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        input_neuron=cfg['DATALOADER']['INPUT_NEURON'],
        uncertainty=cfg['DATALOADER']['UNCERTAINTY'],
        transition_probability=cfg['DATALOADER']['TRANSITION_PROB'],
        g_min=cfg['DATALOADER']['G_MIN'],
        g_max=cfg['DATALOADER']['G_MAX'],
        sigma_sq=cfg['DATALOADER']['SIGMA_SQ'],
        batch_size=cfg['TRAIN']['BATCHSIZE'],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
        num_workers=4, shuffle=True,
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

    model.train()
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        for i, data in enumerate(train_dataloader):
            inputs, true_signal_list, _, _ = data
            inputs = Variable(inputs.float()).to(device)
            true_signal_list = Variable(true_signal_list.float()).to(device)

            hidden_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE']))
            hidden = torch.from_numpy(hidden_np).float()
            hidden = hidden.to(device)

            reservoir_np = np.random.normal(0, 0.5, size=(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['RESERVOIR']))
            reservoir = torch.from_numpy(reservoir_np).float()
            reservoir = reservoir.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            reservoir = reservoir.detach()

            if cfg['MODEL']['RESERVOIR'] == 0:
                _, output_list = model(
                    inputs,
                    hidden,
                    cfg['DATALOADER']['TIME_LENGTH'],
                )
            else:
                _, output_list, _ = model(
                    inputs,
                    hidden,
                    reservoir,
                    cfg['DATALOADER']['TIME_LENGTH'],
                )

            true_value_loss = 0
            for i in range(cfg['TRAIN']['BATCHSIZE']):
                for t in range(20, 120):
                    true_value_loss += torch.nn.MSELoss()(
                        output_list[i, t, 0],
                        true_signal_list[i, t],
                    )
            true_value_loss.backward()
            optimizer.step()

            if cfg['MODEL'].get('TRAIN_ALPHA', False):
                if cfg['MODEL']['RESERVOIR'] == 0:
                    model.alpha.data.clamp_(0, 1)
                else:
                    model.alpha_fast.data.clamp_(0, 1)
                    model.alpha_slow.data.clamp_(0, 1)

        if epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            print('true_signal: ', true_signal_list[0, -10:].detach().cpu().numpy())
            print('output: ', output_list[0, -10:, 0].detach().cpu().numpy())
            print(f'Train Epoch: {epoch}, TrueValueLoss: {true_value_loss.item() / 5000:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
