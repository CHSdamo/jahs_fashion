
import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from cnn import torchModel
from utils import early_stopping


def train_optim(args):
    with open(args.working_dir + '/opt_cfg.json') as json_file:
        opt_cfg = json.load(json_file)

    print(opt_cfg)
    max_epochs = 50
    seed = opt_cfg['seed']

    lr = opt_cfg['learning_rate_init'] if opt_cfg['learning_rate_init'] else 0.001
    weight_decay = opt_cfg['weight_decay'] if opt_cfg['weight_decay'] else 0
    batch_size = opt_cfg['batch_size'] if opt_cfg['batch_size'] else 200

    data_dir = opt_cfg['data_dir']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    model_device = torch.device(device)

    img_width = 28
    img_height = 28
    input_shape = (1, img_width, img_height)

    trans = [transforms.ToTensor()]
    trans.append(transforms.RandomHorizontalFlip()) if opt_cfg['random_horizontal_flip'] else trans
    trans.append(transforms.RandomRotation(10)) if opt_cfg['random_rotation'] else trans

    pre_processing = transforms.Compose(trans)

    train_val = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=pre_processing
    )

    test = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    model = torchModel(opt_cfg,
                       input_shape=input_shape,
                       num_classes=len(train_val.classes)).to(model_device)

    if opt_cfg['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_criterion = torch.nn.CrossEntropyLoss().to(model_device)

    train_loader = DataLoader(dataset=train_val,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test,
                             batch_size=batch_size,
                             shuffle=False)
    score_lst = []
    best_score = None
    counter = 0
    for epoch in range(max_epochs):
        logging.info('Full Test Start:')
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, max_epochs))
        train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, model_device)
        logging.info('Train accuracy %f', train_score)
        val_loss = model.validation(test_loader, train_criterion, model_device, best_score, )
        best_score, counter, early_stop = early_stopping(val_loss, model, best_score, counter)

        if early_stop:
            print("Early stopping")
            break
        score_lst.append(train_score)

    test_score = model.test(test_loader, model_device)
    score_lst.append(test_score)
    print(f'Test accuracy: {test_score}')
    with open(args.working_dir + '/score_lst.json', 'w') as f:
        json.dump(score_lst, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JAHS')

    parser.add_argument('--working_dir', default='./tmp_exp', type=str,
                        help="directory where intermediate results are stored")

    args = parser.parse_args()

    train_optim(args)