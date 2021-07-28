import math
import h5py
import torch
import torch.nn as nn
from torch import Tensor
from Scripts.transformer.util import tgt_unpadding_mask_extention

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device() # google colab tpu
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size),device=DEVICE)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:]).to(DEVICE)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(DEVICE)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return (self.embedding(tokens.long()) * math.sqrt(self.emb_size)).to(DEVICE)


class SenEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(SenEmbedding, self).__init__()
        self.emb_size = emb_size
        self.PadEmbedding = torch.randn(self.emb_size, device=DEVICE)
    def forward(self, tgt: Tensor, PAD_IDX: int):
        '''
        :param tgt: (max number of sentences in stories, batch size)
        :param PAD_IDX: 1
        :return: (max number of sentences in stories, batch size, dimension of sentence embedding)
        '''

        tgt_emb = torch.stack(
            [torch.randn(self.emb_size,device=DEVICE) if int(index.item()) != PAD_IDX else self.PadEmbedding for _ in tgt for index in _])
        return tgt_emb.view(tgt.size()[0],tgt.size()[1],-1).to(DEVICE)


class TokenSenEmbedding(nn.Module):
    def __init__(self, src_vocab_size, emb_size):
        super(TokenSenEmbedding, self).__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size).to(DEVICE)
        self.emb_size = emb_size
    def forward(self, src: Tensor, SenEmbedding_dict_file_path: str):
        '''
        :param src: (max number of tokens in plots + max number of sentences in stories, batch size)
        :param SenEmbedding_dict_file_path: filr path of SenEmbedding dictionary
        :return: (max number of tokens in plots + max number of sentences in stories, batch size, dimension of sentence embedding)
        '''

        output = torch.stack(
            [self.src_tok_emb(index) if index < 10000000 else torch.tensor(h5py.File(
                SenEmbedding_dict_file_path,"r")[str(int(index.item()))][...]) for _ in src for index in _])
        return output.view(src.size()[0],src.size()[1],-1).to(DEVICE)



class SenEmbedding_Loss(nn.Module):
    def __init__(self):
        super(SenEmbedding_Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Tensor, is_pad: Tensor, tgt_out: Tensor, tgt_padding_mask: Tensor,
                lambda1: float, lambda2: float, lambda3: float, lambda4: float, lambda5: float):
        '''
        :param logits: (max number of sentences in stories, batch size, dimension of sentence embedding)
        :param is_pad: (max number of sentences in stories, batch size, 1)
        :param tgt_out: (max number of sentences in stories, batch size, dimension of sentence embedding)
        :param tgt_padding_mask: (batch size, max number of sentences in stories), masked position == True, else == False
        :return: a float loss figure in a tensor, tensor([])
        '''
        unpadding_matrix = tgt_unpadding_mask_extention(tgt_padding_mask, tgt_out.size()[2]).to(DEVICE)
        # torch.masked_select(input, mask, *, out=None) returns a new 1-D tensor which indexes the input tensor
        # according to the boolean mask (where is True) which is a BoolTensor.
        logits_flattened = torch.masked_select(logits, unpadding_matrix).to(DEVICE)
        tgt_out_flattened = torch.masked_select(tgt_out, unpadding_matrix).to(DEVICE)

        # MSE Loss: prediction vs gold
        loss_mse = self.mse(logits_flattened, tgt_out_flattened)

        # Similarity Loss: prediction vs gold
        logits_flattened = logits_flattened.view(-1, tgt_out.size()[2])
        tgt_out_flattened = tgt_out_flattened.view(-1, tgt_out.size()[2])
        tgt_label = torch.ones(tgt_out_flattened.size()[0],device=DEVICE)
        loss_cos = self.cos(logits_flattened, tgt_out_flattened, tgt_label)

        # Delta Loss: predicted sentence 1 vs predicted sentence 2
        delta_list_logits = delta_list(logits, tgt_padding_mask).to(DEVICE)
        delta_list_tgt_out = delta_list(tgt_out, tgt_padding_mask).to(DEVICE)
        loss_delta = self.mse(delta_list_logits,delta_list_tgt_out )

        # Delta of delta loss: delta 1 vs delta 2
        loss_delta_of_delta = (self.mse(delta_of_delta_list(delta_list_logits), delta_of_delta_list(delta_list_tgt_out))/1000                                     )

        # BCE Loss: is pad sentence embedding or not
        is_pad = is_pad.view(is_pad.size()[0],is_pad.size()[1]).to(DEVICE)
        zero = torch.zeros_like(is_pad.transpose(0,1),device=DEVICE)
        one = torch.ones_like(is_pad.transpose(0,1),device=DEVICE)
        # tgt_padding_mask (length of sentence, batch size)
        tgt_padding_mask = torch.where(tgt_padding_mask == False, zero, one).transpose(0,1).to(DEVICE)
        loss_bce = self.bce(is_pad, tgt_padding_mask)

        loss = lambda1 * loss_mse + lambda2 * loss_cos + lambda3 * loss_delta + lambda4 * loss_delta_of_delta + lambda5 * loss_bce

        #print("loss_mse",loss_mse)
        #print("loss_cos",loss_cos)
        #print("loss_delta",loss_delta)
        #print("loss_delta_of_delta",loss_delta_of_delta)
        #print("loss_bce",loss_bce)
        return loss



def delta_list(input: Tensor, tgt_padding_mask: Tensor):
    '''
    :param input: (max number of sentences in stories, batch size, dimension of sentence embedding)
    :param tgt_padding_mask: (batch size, max number of sentences in stories), masked position == True, else == False
    :return: a 1D float tensor of similarity between each sentence and the next with padding sentences excluded
    '''
    sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    input = input.transpose(0, 1).type(torch.DoubleTensor)
    deltas = []
    for n_batch, batch in enumerate(input):
        for n_sen in range(batch.size()[0] - 1):
            if (tgt_padding_mask[n_batch][n_sen].item() or tgt_padding_mask[n_batch][n_sen + 1].item()) is False:
                delta = sim(batch[n_sen:n_sen + 1], batch[n_sen + 1:n_sen + 2]).item()
                deltas.append(delta)
    return torch.tensor(deltas).to(DEVICE)


def delta_of_delta_list(delta_list: Tensor):
    '''
    :param delta_list: a 1D float tensor of similarity between each sentence and the next with padding sentences excluded
    :return: a 1D float tensor of changing rate between each similarity score with padding sentences excluded
    '''
    delta_of_delta_list = []
    for n in range(len(delta_list) - 1):
        if delta_list[n].item() != 0:
            delta_of_delta = (delta_list.to(DEVICE)[n + 1].item() - delta_list[n].item()) / delta_list[n].item()
        else:
            delta_of_delta = (delta_list.to(DEVICE)[n + 1].item() - delta_list[n].item()) / 1e-6
        delta_of_delta_list.append(delta_of_delta)
    return torch.tensor(delta_of_delta_list).to(DEVICE)