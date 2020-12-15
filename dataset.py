import pandas as pd

import torch
from torch.utils.data import Dataset

from alphabets import Protein, Chem


class ProteinChemicalDataset(Dataset):
    def __init__(self, mode='train'):
        data = pd.read_csv('./data/preprocessed_data.tsv', sep='\t', header=None)
        self.labels = data[0]
        self.proteins = data[1]
        self.molecules = data[2]

        self.protein_encoder = Protein()
        self.chem_encoder = Chem()

    def __len__(self):
        return len(self.labels)

    def _preprocess(self, x0, x1=None, special_idx=21, max_len=512):
        """ pre-processing steps for PLUS-TFM pre-training """
        special_tokens = {"MASK": torch.tensor([special_idx], dtype=torch.long),
                          "CLS": torch.tensor([special_idx + 1], dtype=torch.long),
                          "SEP": torch.tensor([special_idx + 2], dtype=torch.long)}
        tokens = torch.zeros(max_len, dtype=torch.long)
        segments = torch.zeros(max_len, dtype=torch.long)
        input_mask = torch.zeros(max_len, dtype=torch.bool)

        # -3  for special tokens [CLS], [SEP], [SEP]
        x0, x1 = self._truncate_seq_pair(x0, x1, max_len)
        x0, x1 = torch.tensor(x0, dtype=torch.long), torch.tensor(x1, dtype=torch.long)

        # set tokens and segments
        if x1 is not None:
            length = len(x0) + len(x1) + 3
            tokens[:length] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"], x1, special_tokens["SEP"]])
            segments[len(x0) + 2:length] = 1
            input_mask[:length] = True
        else:
            length = len(x0) + 2
            tokens[:len(x0) + 2] = torch.cat([special_tokens["CLS"], x0, special_tokens["SEP"]])
            input_mask[:len(x0) + 2] = True

        return tokens, segments, input_mask, length

    @staticmethod
    def _truncate_seq_pair(x0, x1, max_len):
        """ clip sequences for the maximum length limitation """
        if x1 is not None:
            max_len -= 3
            while True:
                if len(x0) + len(x1) <= max_len:
                    break
                elif len(x0) > len(x1):
                    x0 = x0[:-1]
                else:
                    x1 = x1[:-1]
        else:
            max_len -= 2
            x0 = x0[:max_len]
        return x0, x1

    def __getitem__(self, idx):
        label, protein, chem = self.labels[idx], self.proteins[idx], self.molecules[idx]
        label = torch.tensor(label, dtype=torch.long)
        protein = self.protein_encoder.encode(bytes(protein, encoding='utf8'))
        chem = self.chem_encoder.encode(bytes(chem, encoding='utf8'))
        tokens, segments, input_mask, length = self._preprocess(protein, chem)
        return tokens, segments, input_mask, length, label




