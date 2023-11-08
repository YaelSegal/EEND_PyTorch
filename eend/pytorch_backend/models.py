# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import nemo.collections.asr as nemo_asr
from math import ceil

class NoamScheduler(_LRScheduler):
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

        # the initial learning rate is set as step = 1
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
            self.last_epoch = 0
        print(self.d_model)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_speakers,
        in_size,
        n_heads,
        n_units,
        n_layers,
        dim_feedforward=2048,
        dropout=0.5,
        decode=True,
        has_pos=False,
    ):
        """Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.decoder = nn.Linear(n_units, n_speakers)
        self.decode = decode

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, ilens, has_mask=False, activation=None):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # ilens = [x.shape[0] for x in src]
        # src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)
        if self.decode:
            # output: (B, T, C)
            output = self.decoder(output)

            if activation:
                output = activation(output)

            output = [out[:ilen] for out, ilen in zip(output, ilens)]

            return output
        else:
            return output

    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []

        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])

        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class EncoderDecoderAttractor(nn.Module):
    def __init__(self, n_units, encoder_dropout=0.1, decoder_dropout=0.1, bidirectional=False):
        super(EncoderDecoderAttractor, self).__init__()

        self.encoder = nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.counter = nn.Sequential(nn.Linear(n_units, 1), nn.Sigmoid())
        self.n_units = n_units

    def forward(self, xs, zeros, lens=None):
        if lens is None:
            output, (h_e, c_e) = self.encoder(packed)  # default (h, c) are zeros
            attractors, (h_d, c_d) = self.decoder(zeros, (h_e, c_e))
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(xs, lens, batch_first=True, enforce_sorted=False)
            output, (h_e, c_e) = self.encoder(packed)  # default (h, c) are zeros
            attractors, (h_d, c_d) = self.decoder(zeros, (h_e, c_e))
        attractors_prob = self.counter(attractors)
        return attractors, attractors_prob


def create_speakers_zeros(batch_size, max_speakers, n_units, device):
    return torch.zeros(batch_size, max_speakers, n_units, device=device)


class TransformerEDADiarization(nn.Module):
    def __init__(
        self,
        in_size,
        n_units,
        n_heads,
        n_layers,
        dropout,
        attractor_encoder_dropout=0.1,
        attractor_decoder_dropout=0.1,
        has_pos=False,
        shuffle=False,
    ):
        """Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
          has_pos (bool): Whether to use positional encoding
        """
        super(TransformerEDADiarization, self).__init__()

        self.enc = TransformerModel(
            n_speakers=15,
            in_size=in_size,
            n_units=n_units,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            has_pos=has_pos,
            decode=False,
        )
        self.eda = EncoderDecoderAttractor(
            n_units,
            encoder_dropout=attractor_encoder_dropout,
            decoder_dropout=attractor_decoder_dropout,
        )
        self.shuffle = shuffle

    def forward(self, xs, n_speakers=None, activation=None):
        lens = [x.shape[0] for x in xs]

        emb = self.enc(xs, lens)
        zeros = create_speakers_zeros(
            len(lens), max(n_speakers) + 1, emb[0].shape[1], emb.device
        )  # add one for the next zero attractor
        if self.shuffle and self.training:
            orders = [
                torch.arange(max(lens)) for i, e_len in enumerate(lens)
            ]  # in training all segment has the same length therefore we can shuffle the order

            for idx, (order, c_len) in enumerate(zip(orders, lens)):
                new_per = torch.randperm(c_len)
                order[:c_len] = new_per
            stack_orders = torch.stack(orders)
            attractors, attractors_prob = self.eda(
                emb[torch.arange(emb.shape[0]).unsqueeze(1), stack_orders], zeros, lens
            )
        else:
            attractors, attractors_prob = self.eda(emb, zeros, lens)
        # ys = [F.matmul(e, att, transb=True) for e, att in zip(emb, attractors)]
        logits = torch.matmul(emb, attractors.transpose(1, 2))
        return logits, attractors_prob

    def estimate(self, xs, lens=None, n_speakers=15, threshold=0.5):
        lens = [x.shape[0] for x in xs]
        emb = self.enc(xs, lens)
        zeros = create_speakers_zeros(
            len(lens), n_speakers + 1, emb[0].shape[1], emb.device
        )  # add one for the next zero attractor
        if self.shuffle and self.training:
            orders = [
                torch.arange(max(lens)) for i, e_len in enumerate(lens)
            ]  # in training all segment has the same length therefore we can shuffle the order

            for idx, (order, c_len) in enumerate(zip(orders, lens)):
                new_per = torch.randperm(c_len)
                order[:c_len] = new_per
            stack_orders = torch.stack(orders)
            attractors, attractors_prob = self.eda(
                emb[torch.arange(emb.shape[0]).unsqueeze(1), stack_orders], zeros, lens
            )
        else:
            attractors, attractors_prob = self.eda(emb, zeros, lens)
        # ys = [F.matmul(e, att, transb=True) for e, att in zip(emb, attractors)]
        logits = torch.matmul(emb, attractors.transpose(1, 2))
        non_active_speaker = attractors_prob < threshold
        logits = logits.transpose(1, 2)
        logits[non_active_speaker.squeeze(2)] = -torch.inf
        return logits.transpose(1, 2), attractors_prob


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TitanetEDADiarization(nn.Module):
    def __init__(
        self,
        n_units,
        context=1,
        attractor_encoder_dropout=0.1,
        attractor_decoder_dropout=0.1,
        shuffle=False,
        freeze_encoder=False,
    ):
        """Self-attention-based diarization model.

        Args:
          n_units (int): Number of units in a self-attention block
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        super(TitanetEDADiarization, self).__init__()

        self.enc = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
        if freeze_encoder:
            for param in self.enc.parameters():
                param.requires_grad = False
        else:
            for param in self.enc.parameters():
                param.requires_grad = True
        self.contector = nn.Sequential(
            nn.Linear(3072, n_units),
            LambdaLayer(lambda x: x.transpose(1, 2)),
            nn.AvgPool1d(kernel_size=context),
            LambdaLayer(lambda x: x.transpose(1, 2)),
        )  # check input dim!
        self.context = context
        self.eda = EncoderDecoderAttractor(
            n_units,
            encoder_dropout=attractor_encoder_dropout,
            decoder_dropout=attractor_decoder_dropout,
        )
        self.shuffle = shuffle

    def enc_foward(self, xs, lens=None):
        processed_signal, processed_signal_len = self.enc.preprocessor(
            input_signal=xs,
            length=torch.LongTensor(lens).to(xs.device) if lens is not None else None,
        )
        encoded, length = self.enc.encoder(audio_signal=processed_signal, length=processed_signal_len)
        return encoded, length

    def forward(self, xs, xlens, n_speakers=None, activation=None):
        emb, new_lengths = self.enc_foward(xs, xlens)
        emb = self.contector(emb.transpose(1, 2))
        lens = [ceil(leni.item() /2) for leni in new_lengths]
        emb = emb[:, :max(lens), :]
        zeros = create_speakers_zeros(
            len(lens), max(n_speakers) + 1, emb[0].shape[1], emb.device
        )  # add one for the next zero attractor
        if self.shuffle and self.training:
            orders = [
                torch.arange(max(lens)) for i, e_len in enumerate(lens)
            ]  # in training all segment has the same length therefore we can shuffle the order

            for idx, (order, c_len) in enumerate(zip(orders, lens)):
                new_per = torch.randperm(c_len)
                order[:c_len] = new_per
            stack_orders = torch.stack(orders)
            attractors, attractors_prob = self.eda(
                emb[torch.arange(emb.shape[0]).unsqueeze(1), stack_orders], zeros, lens
            )
        else:
            attractors, attractors_prob = self.eda(emb, zeros, lens)
        # ys = [F.matmul(e, att, transb=True) for e, att in zip(emb, attractors)]
        logits = torch.matmul(emb, attractors.transpose(1, 2))
        return logits, attractors_prob

    def estimate(self, xs, xlens, n_speakers=15, threshold=0.5):
        emb, new_lengths = self.enc_foward(xs, xlens)
        emb = self.contector(emb.transpose(1, 2))
        lens = [ceil(leni.item() /2) for leni in new_lengths]

        zeros = create_speakers_zeros(
            len(lens), n_speakers + 1, emb[0].shape[1], emb.device
        )  # add one for the next zero attractor
        if self.shuffle and self.training:
            orders = [
                torch.arange(max(lens)) for i, e_len in enumerate(lens)
            ]  # in training all segment has the same length therefore we can shuffle the order

            for idx, (order, c_len) in enumerate(zip(orders, lens)):
                new_per = torch.randperm(c_len)
                order[:c_len] = new_per
            stack_orders = torch.stack(orders)
            attractors, attractors_prob = self.eda(
                emb[torch.arange(emb.shape[0]).unsqueeze(1), stack_orders], zeros, lens
            )
        else:
            attractors, attractors_prob = self.eda(emb, zeros, lens)
        # ys = [F.matmul(e, att, transb=True) for e, att in zip(emb, attractors)]
        logits = torch.matmul(emb, attractors.transpose(1, 2))
        non_active_speaker = attractors_prob < threshold
        logits = logits.transpose(1, 2)
        logits[non_active_speaker.squeeze(2)] = -torch.inf
        return logits.transpose(1, 2), attractors_prob


if __name__ == "__main__":
    import torch

    model = TransformerModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
