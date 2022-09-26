"""
===========================
Optimization using BOHB
===========================
"""
import argparse
import json
import logging

import numpy as np
import torch

from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import Constant
from sklearn.model_selection import StratifiedKFold

from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms

from cnn import torchModel
from search_space import search_space
from datetime import datetime
from train_optim import train_optim


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_from_cfg(cfg: Configuration, seed: int, instance: str, budget: float):
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param instance: str
        used to represent the instance to use (just a placeholder for this example)
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
    weight_decay = cfg['weight_decay'] if cfg['weight_decay'] else 0
    batch_size = cfg['batch_size'] if cfg['batch_size'] else 200

    data_dir = cfg['data_dir'] if cfg['data_dir'] else 'FashionMNIST'
    device = cfg['device'] if cfg['device'] else 'cpu'

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    # input processing
    img_width = 28
    img_height = 28
    # scale = cfg['image_scale'] if cfg['image_scale'] else 1
    # img_width = int(img_width * scale)
    # img_height = int(img_height * scale)
    input_shape = (1, img_width, img_height)

    trans = [transforms.ToTensor()]
    trans.append(transforms.RandomHorizontalFlip()) if cfg['random_horizontal_flip'] else trans
    trans.append(transforms.RandomRotation(10)) if cfg['random_rotation'] else trans

    pre_processing = transforms.Compose(trans)

    train_val = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=pre_processing
    )

    # returns the cross validation accuracy
    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent
    num_epochs = int(np.ceil(budget))
    score = []

    # for train_idx, valid_idx in cv.split(data, data.targets):
    for train_idx, valid_idx in cv.split(train_val, train_val.targets):
        train_data = Subset(train_val, train_idx)
        val_data = Subset(train_val, valid_idx)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)
        model = torchModel(cfg,
                           input_shape=input_shape,
                           num_classes=len(train_val.classes)).to(model_device)

        summary(model, input_shape, device=device)

        print('Batch size: {}'.format(cfg['batch_size']))
        print('Optimizer: {}'.format(cfg['optimizer']))
        print('Kernel size: {}x{}'.format(cfg['kernel_size'], cfg['kernel_size']))
        print('Batch norm: {}'.format(cfg['use_BN']))
        print('global_avg_pooling: {}'.format(cfg['global_avg_pooling']))
        print('learning_rate_init: {}'.format(cfg['learning_rate_init']))
        print('weight_decay: {}'.format(cfg['weight_decay']))
        print('dropout_rate: {}'.format(cfg['dropout_rate']))
        print('Random_horizontal_flip: {}'.format(cfg['random_horizontal_flip']))
        print('Random_rotation: {}'.format(cfg['random_rotation']))

        train_criterion = torch.nn.CrossEntropyLoss

        if cfg['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif cfg['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_criterion = train_criterion().to(device)

        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, model_device)
            logging.info('Train accuracy %f', train_score)
            print('Train loss:', train_loss)
            print('Train accuracy:', train_score)

        val_score = model.eval_fn(val_loader, device)
        score.append(val_score)

    val_acc = 1 - np.mean(score)  # because minimize
    print('Val_acc:', val_acc)
    return val_acc


if __name__ == '__main__':
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the forbidden clauses.
    """
    parser = argparse.ArgumentParser(description='JAHS')

    parser.add_argument('--runtime', default=21600, type=int, help='Running time allocated to run the algorithm')

    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--working_dir', default='./tmp_exp', type=str,
                        help="directory where intermediate results are stored")

    parser.add_argument('--max_epochs', type=int, default=20, help='maximal number of epochs to train the network')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='device to run the models')

    args = parser.parse_args()

    logger = logging.getLogger("JAHS_fashion")
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    cs = search_space()

    data_dir = args.data_dir
    runtime = args.runtime
    device = args.device
    max_epochs = args.max_epochs
    working_dir = args.working_dir
    seed = args.seed

    cs.add_hyperparameters([
        Constant('device', device),
        Constant('data_dir', data_dir),
        Constant('seed', seed)
    ])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",        # we optimize quality (alternative to runtime)
                         "wallclock-limit": runtime,  # max duration to run the optimization (in seconds)
                         "cs": cs,                    # configuration search space
                         'output-dir': working_dir,   # working directory where intermediate results are stored
                         "deterministic": "True",
                         })

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the CNN for
    # intensifier parameters (Budget parameters for BOHB)
    intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_epochs, 'eta': 3}

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(scenario=scenario, rng=np.random.RandomState(seed),
                   tae_runner=cnn_from_cfg,
                   intensifier_kwargs=intensifier_kwargs,
                   # all arguments related to intensifier can be passed like this
                   initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
                                          'max_config_fracs': .2})

    # Start optimization
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=max_epochs, seed=seed)[1]
    print("Optimized Value: %.4f" % inc_value)

    # store your optimal configuration to disk
    opt_config = incumbent.get_dictionary()
    with open(args.working_dir + '/opt_cfg.json', 'w') as f:
        json.dump(opt_config, f)

    # Run a full training-test-process
    train_optim(args)
