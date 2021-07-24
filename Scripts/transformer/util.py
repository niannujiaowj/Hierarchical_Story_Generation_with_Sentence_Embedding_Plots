import torch
from torch import Tensor


# count the number of lines in a file
def count_line(file_name):
    with open(file_name) as f:
        for count, _ in enumerate(f, 1):
            pass
    return count


# get gold target output of sentence embeddings
def Tgt_Out(tgt: Tensor, tgt_SenEmbedding_dict_file_path: str):
    '''
    :param tgt: tensor (max number of sentences in stories, batch size)
    :param tgt_SenEmbedding_dict: file path
    :return: tensor (max number of sentences in stories - 1, batch size, dimension of sentence embedding)
    '''

    import h5py
    PadSenEmbedding = torch.randn(len(h5py.File(tgt_SenEmbedding_dict_file_path,"r")["10000000"][...]))

    # tgt[1:] is to exclude <bos> token
    tgt_out = torch.stack(
        [PadSenEmbedding if index.item() == 1 else torch.tensor(h5py.File(
            tgt_SenEmbedding_dict_file_path, "r")[str(index.item())][...]) for _ in tgt[1:] for index in _])
    '''tgt_out = [[] for _ in range(tgt.size()[0]-1)] # exclude <bos> token
    for n, sen in enumerate(tgt[1:]): # exclude <bos> token
        for index in sen:
            if index.tolist() == 1:
                tgt_out[n].append(PadSenEmbedding.tolist())
            else:
                tgt_out[n].append(tgt_SenEmbedding_dict[index.tolist()].tolist())'''
    return tgt_out.view(tgt.size()[0]-1,tgt.size()[1],-1)



# get the unpadding matrix
def tgt_unpadding_mask_extention(tgt_padding_mask:Tensor, emb_size: int):
    '''
    :param tgt_padding_mask: (batch size, max number of sentences in stories), masked position == True, else == False
    :param emb_size: default 768
    :return: boolean tensor (max number of sentences in stories, batch size, dimension of sentence embedding)
            unpadding = True, padding = False
    '''
    tgt_unpadding_mask_extention = ~torch.stack(
        [torch.zeros(emb_size, dtype=torch.bool) if index.item() == False else torch.ones(emb_size, dtype=torch.bool)
         for _ in tgt_padding_mask for index in _])
    return tgt_unpadding_mask_extention.view(tgt_padding_mask.size()[1],tgt_padding_mask.size()[0],-1)

    '''tgt_unpadding_mask_extention = torch.zeros(tgt_padding_mask.size()[1],tgt_padding_mask.size()[0],emb_size)
    for n_sen, sen in  enumerate(tgt_unpadding_mask_extention):
        for n_batch, batch in enumerate(sen):
            tgt_unpadding_mask_extention[n_sen:n_sen+1,n_batch:n_batch+1,:] = \
                tgt_padding_mask.transpose(0,1)[n_sen:n_sen+1,n_batch:n_batch+1]
    return ~tgt_unpadding_mask_extention.to(torch.bool)'''




