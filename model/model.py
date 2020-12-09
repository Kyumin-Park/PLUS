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

    def forward(self, tokens, segments, input_mask, masked_pos=None, per_seq=True, embedding=False):
        h = self.transformer(tokens, segments, input_mask)

        if embedding:
            return h
        else:
            logits_lm = None
            if masked_pos is not None:
                masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
                h_masked = torch.gather(h, 1, masked_pos)
                h_masked = self.norm_lm(tfm.gelu(self.fc_lm(h_masked)))
                logits_lm = F.log_softmax(self.decoder(h_masked) + self.decoder_bias, dim=2)

            if per_seq:
                logits_cls = self.cls(self.drop_cls(h[:, 0]))
            else:
                logits_cls = self.cls(self.drop_cls(h))

            return logits_lm, logits_cls

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


def get_loss(batch, models_dict, cfg, tasks_dict, args, test=False):
    """ feed-forward and evaluate PLUS_TFM model """
    models, models_idx, tasks_idx = models_dict["model"], models_dict["idx"], tasks_dict["idx"]
    if not test: tasks_flag = tasks_dict["flags_train"]
    else:        tasks_flag = tasks_dict["flags_eval"]
    per_seq = True
    tokens, segments, input_mask, labels = batch
    masked_pos = None

    logits_lm, logits_cls = models[models_idx.index("")](tokens, segments, input_mask, masked_pos, per_seq)

    results = []
    if per_seq: result = args["evaluate_cls"](logits_cls, labels, flag, args)
    else:       result = args["evaluate_cls"](logits_cls, labels, label_weights, flag)
    if cfg.cls_loss_lambda != -1: result["avg_loss"] = result["avg_loss"] * cfg.cls_loss_lambda
    if "aggregate" in args:
        result["valid"] = [valids.cpu()]
        result["label_weights"] = [label_weights.cpu()]

        results.append(result)

    return results


def get_embedding(batch, models_dict, args):
    """ feed-forward and evaluate PLUS_TFM model """
    models, models_idx = models_dict["model"], models_dict["idx"]
    tokens, segments, input_mask = batch

    model = models[models_idx.index("")]
    h = model(tokens, segments, input_mask, embedding=True)

    h_list = model.module.em(h, input_mask, cpu=True) if args["data_parallel"] else model.em(h, input_mask, cpu=True)
    embeddings = [[], h_list]

    return embeddings


