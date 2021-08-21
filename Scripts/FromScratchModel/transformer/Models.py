import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import time
from Scripts.MyModel.transformer import PositionalEncoding, TokenEmbedding, SenEmbedding, TokenSenEmbedding
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device() # google colab tpu
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, src_vocab_size: int,
                 tgt_vocab_size: int, dim_feedforward: int, dropout: float, nhead: int, mode: str, batch_size: int):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size,device=DEVICE)
        # !!
        self.mode = mode
        if  self.mode == "WritingPrompts2SenEmbeddings":
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
            self.tgt_tok_emb = SenEmbedding(emb_size)
        else:
            self.src_tok_emb = TokenSenEmbedding(src_vocab_size, emb_size) # ????need detach??????
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.batch_size = batch_size



    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor, PAD_IDX: int, SenEmbedding_dict_file_path: str):
        # !!
        if self.mode == "WritingPrompts2SenEmbeddings":
            start_time = time.time()
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            end_time = time.time()
            print("src_emb", end_time - start_time)

            start_time = time.time()
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt, PAD_IDX))
            end_time = time.time()
            print("tgt_emb", end_time - start_time)
        else:
            src_emb = self.positional_encoding(self.src_tok_emb(src,SenEmbedding_dict_file_path))
            tgt_emb = self.positional_faencoding(self.tgt_tok_emb(tgt))

        start_time = time.time()
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask,
                                        tgt_padding_mask, memory_key_padding_mask)
        end_time = time.time()
        print("outs", end_time - start_time)

        # !!
        if self.mode == "WritingPrompts2SenEmbeddings":
            return outs, self.generator(outs)
        else:
            return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, PAD_IDX: int):
        # !!
        if self.mode == "WritingPrompts2SenEmbeddings":
            return self.transformer.decoder(self.positional_encoding(
                self.tgt_tok_emb(tgt, PAD_IDX)), memory,
                tgt_mask)
        else:
            return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)










