import os
import torch
from torch import Tensor


# count the number of lines in a file
def count_line(file_name):
    with open(file_name) as f:
        for count, _ in enumerate(f, 1):
            pass
    return count

# get pad positions
"""def pad_positions(tgt_tensor: Tensor, pad_index: int):
    '''
    :param tgt_tensor: (max number of sentences in stories, batch size)
    :pad_index: int
    :return: a list of position pairs [(0,0),(0,10),...,(166,128)]
    '''
    pad_positions = []
    for n_sen, sen in enumerate(tgt_tensor):
        for n_batch, batch in enumerate(sen):
            if batch.tolist() == pad_index:
                pad_positions.append((n_sen, n_batch))
    return pad_positions"""


# get gold target output when generating sentence embeddings
def Tgt_Out(tgt: Tensor, tgt_SenEmbedding_dict: str):
    '''
    :param tgt: tensor (max number of sentences in stories, batch size)
    :param tgt_SenEmbedding_dict: file path
    :return: tensor (max number of sentences in stories, batch size, dimension of sentence embedding)
    '''

    tgt_SenEmbedding_dict = torch.load(tgt_SenEmbedding_dict)
    PadSenEmbedding = torch.randn(len(tgt_SenEmbedding_dict[100000]))
    tgt_out = [[] for _ in range(tgt.size()[0])]
    for n, sen in enumerate(tgt):
        for index in sen:
            if index.tolist() == 1:
                tgt_out[n].append(PadSenEmbedding.tolist())
            else:
                tgt_out[n].append(tgt_SenEmbedding_dict[index.tolist()].tolist())
    return torch.tensor(tgt_out)


# get the unpadding matrix
def tgt_unpadding_mask_extention(tgt_padding_mask:Tensor, emb_size: int):
    '''
    :param tgt_padding_mask: tensor (batch size, max number of sentences in stories)
    :param emb_size: default 768
    :return: boolean tensor (max number of sentences in stories, batch size, dimension of sentence embedding)
            unpadding = True, padding = False
    '''
    tgt_unpadding_mask_extention = torch.zeros(tgt_padding_mask.size()[1],tgt_padding_mask.size()[0],emb_size)
    for n_sen, sen in  enumerate(tgt_unpadding_mask_extention):
        for n_batch, batch in enumerate(sen):
            tgt_unpadding_mask_extention[n_sen:n_sen+1,n_batch:n_batch+1,:] = \
                tgt_padding_mask.transpose(0,1)[n_sen:n_sen+1,n_batch:n_batch+1]
    return ~tgt_unpadding_mask_extention.to(torch.bool)

def save_data(data, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    variable_name = [k for k, v in locals().items() if v == data][0]
    print(variable_name)
    print("Saving {} ...".format(variable_name))
    torch.save(data, save_path + "/{}.pt".format(variable_name))
    print("Data Saved in {}/{}.pt".format(save_path,variable_name))

