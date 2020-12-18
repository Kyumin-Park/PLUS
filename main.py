import os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import ProteinChemBERT, ProteinECFPModel
from dataset import ProteinChemicalDataset, ProteinECFPDataset
from optim import get_BertAdam_optimizer
from train import Trainer


def main():
    cfg = get_args()
    # Configure dataloader
    if cfg.mode == 'ecfp':
        data_cls = ProteinECFPDataset
    elif cfg.mode == 'smiles':
        data_cls = ProteinChemicalDataset
    else:
        raise NotImplementedError

    train_loader = DataLoader(data_cls(mode='train'), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(data_cls(mode='val'), batch_size=32,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(data_cls(mode='test'), batch_size=32,
                             shuffle=False, num_workers=4)

    # Configure extra configs
    cfg.total_steps = len(train_loader) * cfg.epochs

    # Model
    if cfg.mode == 'ecfp':
        model = ProteinECFPModel(cfg).to(cfg.device)
        if cfg.pretrained_model is not None:
            model.bert.load_weights(cfg.pretrained_model)
    elif cfg.mode == 'smiles':
        model = ProteinChemBERT(cfg).to(cfg.device)
        if cfg.pretrained_model is not None:
            model.load_weights(cfg.pretrained_model)
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_BertAdam_optimizer(cfg, model)

    # Run train
    trainer = Trainer(cfg, model, criterion, optimizer,
                      {'train': train_loader, 'val': val_loader})
    trainer.train()

    model.load_state_dict(torch.load(cfg.save_path + 'best_acc.pt', map_location=cfg.device))
    trainer.evaluate(test_loader, 'Test')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='Total Epochs')
    parser.add_argument('--input_dim', default=70, type=int, help='Input dimension (Vocab size)')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=5e-6, type=float, help='Initial Learning Rate')
    parser.add_argument('--warm_up', default=0.1, type=float, help='Warmup steps')
    parser.add_argument('--device', default='cuda:0', type=str, help='Running Device')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--pretrained_model', default='./weight/PLUS-TFM.pt', type=str, help='Pretrained Model')
    parser.add_argument('--val_interval', default=1, type=int, help='Validation interval')
    parser.add_argument('--mode', default='ecfp', type=str, choices=['ecfp', 'smiles'],
                        help='mode of prediction model')
    parser.add_argument('--save_path', default='./weight/checkpoint/', type=str, help='checkpoint path')

    cfg = parser.parse_args()

    with open('config/plus-tfm.json', encoding='utf-8') as f:
        base_cfg = json.load(f)
    for key, item in base_cfg.items():
        setattr(cfg, key, item)

    cfg.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_path, exist_ok=True)

    print(cfg)

    return cfg


if __name__ == '__main__':
    main()
