from tqdm import tqdm

import torch

class Trainer:
    def __init__(self, cfg, model, criterion, optimizer, loader):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = loader['train']
        self.val_loader = loader['val']

    def train(self):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc='Training')
            for idx, data in enumerate(pbar):
                tokens, segments, input_mask, length, label = data
                tokens, segments = tokens.to(self.cfg.device), segments.to(self.cfg.device)
                input_mask, label = input_mask.to(self.cfg.device), label.to(self.cfg.device)

                pred = self.model(tokens, segments, input_mask)
                loss = self.criterion(pred, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.cfg.val_interval == 0 or epoch == (self.cfg.epochs - 1):
                self.validate()

        print('Train Finished')

    def validate(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, total=len(self.val_loader), desc='Validation')
        total_loss, total_acc, n_data = 0, 0, 0
        for idx, data in enumerate(pbar):
            tokens, segments, input_mask, length, label = data
            tokens, segments = tokens.to(self.cfg.device), segments.to(self.cfg.device)
            input_mask, label = input_mask.to(self.cfg.device), label.to(self.cfg.device)

            with torch.no_grad():
                pred = self.model(tokens, segments, input_mask)
            loss = self.criterion(pred, label)

            total_loss += loss.item()
            total_acc += torch.eq(torch.argmax(pred, dim=1), label).item()
            n_data += tokens.size(0)

        print(f'Validation Finished - Average loss: {total_loss / n_data}, Average Acc: {total_acc / n_data}')
