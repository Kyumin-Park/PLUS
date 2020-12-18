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
        best_val = -1
        for epoch in range(self.cfg.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc='Training')
            for idx, data in enumerate(pbar):
                if self.cfg.mode == 'ecfp':
                    tokens, segments, input_mask, length, label, ecfp = data
                elif self.cfg.mode == 'smiles':
                    tokens, segments, input_mask, length, label = data
                else:
                    raise NotImplementedError
                tokens, segments = tokens.to(self.cfg.device), segments.to(self.cfg.device)
                input_mask, label = input_mask.to(self.cfg.device), label.to(self.cfg.device)

                if self.cfg.mode == 'smiles':
                    pred = self.model(tokens, segments, input_mask)
                elif self.cfg.mode == 'ecfp':
                    ecfp = ecfp.to(self.cfg.device)
                    pred = self.model(tokens, segments, input_mask, ecfp)
                else:
                    raise NotImplementedError
                loss = self.criterion(pred, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.cfg.val_interval == 0 or epoch == (self.cfg.epochs - 1):
                val_loss, val_acc = self.evaluate(self.val_loader, 'Validation')
                if val_acc > best_val:
                    torch.save(self.model.state_dict(), self.cfg.save_path + 'best_acc.pt')
        print('Train Finished')

    def evaluate(self, loader, name):
        self.model.eval()
        pbar = tqdm(loader, total=len(loader), desc=name)
        total_loss, total_acc, n_data = 0, 0, 0
        for idx, data in enumerate(pbar):
            if self.cfg.mode == 'ecfp':
                tokens, segments, input_mask, length, label, ecfp = data
            elif self.cfg.mode == 'smiles':
                tokens, segments, input_mask, length, label = data
            else:
                raise NotImplementedError
            tokens, segments = tokens.to(self.cfg.device), segments.to(self.cfg.device)
            input_mask, label = input_mask.to(self.cfg.device), label.to(self.cfg.device)

            with torch.no_grad():
                if self.cfg.mode == 'smiles':
                    pred = self.model(tokens, segments, input_mask)
                elif self.cfg.mode == 'ecfp':
                    ecfp = ecfp.to(self.cfg.device)
                    pred = self.model(tokens, segments, input_mask, ecfp)
                else:
                    raise NotImplementedError
            loss = self.criterion(pred, label)

            total_loss += loss.item()
            total_acc += torch.eq(torch.argmax(pred, dim=1), label).to(torch.int32).sum().item()
            n_data += tokens.size(0)

        avg_loss = total_loss / n_data
        avg_acc = total_acc / n_data
        print(f'{name} Finished - Average loss: {avg_loss}, Average Acc: {avg_acc}')
        return avg_loss, avg_acc

