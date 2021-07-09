import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

from transformer.Modules import PositionalEncoding, TokenEmbedding, SenEmbedding, TokenSenEmbedding





class Transformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, src_vocab_size: int,
                 tgt_vocab_size: int, dim_feedforward: int, dropout: float, NHEAD: int, mode: str, batch_size: int):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        # !!
        self.mode = mode
        if  self.mode == "WritingPrompts2SenEmbeddings":
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
            self.tgt_tok_emb = SenEmbedding(emb_size)
        else:
            self.src_tok_emb = TokenSenEmbedding(src_vocab_size, emb_size)
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.batch_size = batch_size



    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor, PAD_IDX: int, SenEmbedding_dict: dict):
        # !!
        if self.mode == "WritingPrompts2SenEmbeddings":
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt, PAD_IDX))
        else:
            src_emb = self.positional_encoding(self.src_tok_emb(src,SenEmbedding_dict))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        # !!
        if self.mode == "WritingPrompts2SenEmbeddings":
            return outs
        else:
            return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, PAD_IDX: int):
        # !!
        if self.mode == "WritingPrompts2SenEmbeddings":
            return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt, PAD_IDX)), memory,
                          tgt_mask)
        else:
            return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)










