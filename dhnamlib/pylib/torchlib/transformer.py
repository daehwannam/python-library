
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .rnnlib.common import get_indicator


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
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
        >>> pos_encoding = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim] (unless batch_first)
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoding(x)
        """

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x + self.pe[:x.size(0), :]

        if self.batch_first:
            x = x.transpose(0, 1)

        return x


class SelfAttnEncoder(nn.Module):
    def __init__(self,
                 dim_model: int = 512,
                 num_heads: int = 8,
                 dim_hiddens: int = 2048,
                 num_layers: int = 6, 
                 dropout: float = 0.1,
                 # activation="relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first=False):
        super().__init__()

        self.dim_model = dim_model
        self.batch_first = batch_first

        encoder_layer = TransformerEncoderLayer(dim_model, num_heads, dim_hiddens, dropout,
                                                layer_norm_eps=layer_norm_eps,
                                                batch_first=batch_first)
        encoder_layer_norm = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_layer_norm)

    def forward(self, input, padding_mask=None):
        """
        :param input: (Batch, Sequence, Embedding) shaped tensor if batch_first
        :param padding_mask: (Batch, Sequence) shaped binary tensor (regardless batch_first)
                             where padded positions have True values.
        :returns: (Batch, Sequence, Embedding) shaped tensor if batch_first

        """
        if input.dim() != 3:
            shape_format = ('(Batch, Sequence, Embedding)' if self.batch_first else
                            '(Sequence, Batch, Embedding)')
            raise Exception(f'The shape of input should be {shape_format}')

        output = self.transformer_encoder(input, src_key_padding_mask=padding_mask)
        return output


class _Example_SelfAttnEncoder(nn.Module):
    def __init__(self,
                 num_tokens: int,
                 dim_model: int = 512,
                 num_heads: int = 8,
                 dim_hiddens: int = 2048,
                 num_layers: int = 6, 
                 dropout: float = 0.1,
                 # activation="relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first=False):
        super().__init__()

        self.dim_model = dim_model
        self.batch_first = batch_first

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pos_encoding = PositionalEncoding(dim_model, batch_first=batch_first)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.self_attn_encoder = SelfAttnEncoder(
            dim_model, num_heads, dim_hiddens, num_layers, dropout, layer_norm_eps, batch_first)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, padding_mask=None):
        if input.dim() != 2:
            shape_format = ('(Batch, Sequence)' if self.batch_first else
                            '(Sequence, Batch)')
            raise Exception(f'The shape of input should be {shape_format}')

        input = self.embedding(input) * math.sqrt(self.dim_model)
        input = self.dropout_layer(self.pos_encoding(input))
        output = self.self_attn_encoder(input, padding_mask)
        return output


def generate_square_subsequent_mask(size, device=None):
    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _create_transformer_masks(src, tgt, pad_idx, device=None):
    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    #
    # src & tgt --> shape is (Sequence, Batch)

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def get_padding_mask(length_tensor, max_length=None):
    return get_indicator(length_tensor, max_length).squeeze(-1).logical_not()
