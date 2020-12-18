# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Dong-Hyun Lee's code (https://github.com/dhlee347/pytorchic-bert)
# - Tristan Bepler's code (https://github.com/tbepler/protein-sequence-embedding-iclr2019)

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import model.transformer as tfm


class ProteinChemBERT(nn.Module):
    """ PLUS-TFM model """
    def __init__(self, cfg):
        super(ProteinChemBERT, self).__init__()
        self.transformer = tfm.Transformer(cfg)

        # masked language modeling (decoder is shared with embedding layer)
        self.fc_lm = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm_lm = tfm.LayerNorm(cfg)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        # classification
        if cfg.num_classes is not None:
            self.drop_cls = nn.Dropout(cfg.dropout)
            self.cls = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, tokens, segments, input_mask, per_seq=True, embedding=False):
        h = self.transformer(tokens, segments, input_mask)
        if embedding:
            return h

        if per_seq:
            logits_cls = self.cls(self.drop_cls(h[:, 0]))
        else:
            logits_cls = self.cls(self.drop_cls(h))

        return logits_cls

    def load_weights(self, pretrained_model, cls=True):
        # load pre-trained model weights
        state_dict = self.state_dict()
        for key, value in torch.load(pretrained_model, map_location=torch.device('cpu')).items():
            if key.startswith("module"):
                key = key[7:]
            if (cls and state_dict[key].shape == value.shape) or \
                    (not cls and "cls" not in key and state_dict[key].shape == value.shape):
                state_dict[key] = value
        self.load_state_dict(state_dict)

    def em(self, h, input_mask, cpu=False):
        # get representations with different lengths from the collated single matrix
        e = [None] * len(input_mask)
        for i in range(len(input_mask)):
            if cpu: e[i] = h[i, 1:torch.sum(input_mask[i]) - 1].cpu()
            else:   e[i] = h[i, 1:torch.sum(input_mask[i]) - 1]
        return e


class ProteinECFPModel(nn.Module):
    def __init__(self, cfg):
        super(ProteinECFPModel, self).__init__()
        self.bert = ProteinChemBERT(cfg)
        self.chem_linear = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 768)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 2), nn.Softmax(),
        )

    def forward(self, tokens, segments, input_mask, ecfp):
        h = self.bert(tokens, segments, input_mask, embedding=True)
        h = h[:, 0, :]
        chem_feature = self.chem_linear(ecfp)

        features = torch.cat([h, chem_feature], dim=1)
        out = self.classifier(features)
        return out
