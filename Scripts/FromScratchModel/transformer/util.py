import torch
from torch import Tensor
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device() # google colab tpu
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# count the number of lines in a file
def count_line(file_name):
    with open(file_name) as f:
        for count, _ in enumerate(f, 1):
            pass
    return count


# get gold target output of sentence embeddings
def Tgt_Out(tgt: Tensor, tgt_SenEmbedding_dict):
    '''
    :param tgt: tensor (max number of sentences in stories, batch size)
    :param tgt_SenEmbedding_dict: hdf5 file
    :return: tensor (max number of sentences in stories - 1, batch size, dimension of sentence embedding)
    '''

    import h5py
    PadSenEmbedding = torch.randn(768,device=DEVICE)

    # tgt[1:] is to exclude <bos> token
    tgt_out = torch.stack(
        [PadSenEmbedding if int(index.item()) == 1 else torch.tensor(tgt_SenEmbedding_dict[str(int(index.item()))][...],device=DEVICE) for _ in tgt[1:] for index in _])
    return tgt_out.view(tgt.size()[0]-1,tgt.size()[1],-1).to(DEVICE)
    #return torch.randn(tgt.size()[0]-1,tgt.size()[1],768)



# get the unpadding matrix
def tgt_unpadding_mask_extention(tgt_padding_mask:Tensor, emb_size: int):
    '''
    :param tgt_padding_mask: (batch size, max number of sentences in stories), masked position == True, else == False
    :param emb_size: default 768
    :return: boolean tensor (max number of sentences in stories, batch size, dimension of sentence embedding)
            unpadding = True, padding = False
    '''
    tgt_unpadding_mask_extention = ~torch.stack(
        [torch.zeros(emb_size, dtype=torch.bool, device=DEVICE) if index.item() == False else
         torch.ones(emb_size, dtype=torch.bool, device=DEVICE) for _ in tgt_padding_mask for index in _])
    return tgt_unpadding_mask_extention.view(tgt_padding_mask.size()[1],tgt_padding_mask.size()[0],-1).to(DEVICE)




