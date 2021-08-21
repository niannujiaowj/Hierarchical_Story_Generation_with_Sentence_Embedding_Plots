import torch.nn as nn
import torch
from torch import Tensor


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get the unpadding matrix
def tgt_unpadding_mask_extention(labels: Tensor):
    '''
    :param labels: (batch size, max number of sentences in stories), masked position == 1 or 0, else == token number of "<SEN>"
    :return: boolean tensor (max number of sentences in stories, batch size, dimension of sentence embedding)
            unpadding = True, padding = False
    '''
    tgt_unpadding_mask_extention = ~torch.stack(
        [torch.zeros(768, dtype=torch.bool) if (index.item() != 0 and index.item() != 1) else
         torch.ones(768, dtype=torch.bool) for _ in labels for index in _]).to(DEVICE)
    return tgt_unpadding_mask_extention.view(labels.size()[1], labels.size()[0], -1).to(DEVICE)


class SenEmbedding_Loss(nn.Module):
    def __init__(self):
        super(SenEmbedding_Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()
        self.crossentropy = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs_senemb: Tensor, logits: Tensor, labels_senembeddings: Tensor, labels: Tensor,
                lambda1: float = 1.0, lambda2: float = 1.0, lambda3: float = 1.0, lambda4: float = 1.0, lambda5: float = 1.0):
        '''
        :param outputs_senemb: (max number of sentences in stories, batch size, dimension of sentence embedding)
        :param outputs: (batch size, max number of sentences in stories, length of vocabulary)
        :param golden_senemb: (max number of sentences in stories, batch size, dimension of sentence embedding)
        :param labels: (batch size, max number of sentences in stories), masked position == 1 or 0, else == token number of "<SEN>"
        :return: a float loss figure in a tensor, tensor([])
        '''
        unpadding_matrix = tgt_unpadding_mask_extention(labels).to(DEVICE)
        # torch.masked_select(input, mask, *, out=None) returns a new 1-D tensor which indexes the input tensor
        # according to the boolean mask (where is True) which is a BoolTensor.
        outputs_senemb_flattened = torch.masked_select(outputs_senemb, unpadding_matrix).to(DEVICE)
        golden_senemb_flattened = torch.masked_select(labels_senembeddings, unpadding_matrix).to(DEVICE)

        # MSE Loss: prediction vs gold
        loss_mse = torch.sqrt(self.mse(outputs_senemb_flattened, golden_senemb_flattened) + 1e-6)

        # Similarity Loss: prediction vs gold
        outputs_senemb_flattened = outputs_senemb_flattened.view(-1, labels_senembeddings.size()[2]).to(DEVICE)
        golden_senemb_flattened = golden_senemb_flattened.view(-1, labels_senembeddings.size()[2]).to(DEVICE)
        tgt_label = torch.ones(golden_senemb_flattened.size()[0]).to(DEVICE)
        loss_cos = self.cos(outputs_senemb_flattened, golden_senemb_flattened, tgt_label)

        # Delta Loss: predicted sentence 1 vs predicted sentence 2
        delta_list_logits = delta_list(outputs_senemb, labels).to(DEVICE)
        delta_list_tgt_out = delta_list(labels_senembeddings, labels).to(DEVICE)
        loss_delta = self.mse(delta_list_logits, delta_list_tgt_out)

        # Delta of delta loss: delta 1 vs delta 2
        loss_delta_of_delta = self.mse(delta_of_delta_list(delta_list_logits), delta_of_delta_list(delta_list_tgt_out))

        # CrossEntropy Loss: is pad sentence embedding or not
        logits = logits.view(-1, logits.size()[-1]).to(DEVICE)
        labels = labels.flatten().to(DEVICE)
        loss_crossentropy = self.crossentropy(logits, labels)

        loss = lambda1 * loss_mse + lambda2 * loss_cos + lambda3 * loss_delta + lambda4 * loss_delta_of_delta + lambda5 * loss_crossentropy


        #print("loss_mse",loss_mse, "loss_cos",loss_cos, "loss_delta",loss_delta, "loss_delta_of_delta",loss_delta_of_delta, "loss_crossentropy",loss_crossentropy)
        #print("lambda1 * loss_mse",lambda1 * loss_mse, "lambda2 * loss_cos",lambda2 * loss_cos, "lambda3 * loss_delta",lambda3 * loss_delta, " lambda4 * loss_delta_of_delta", lambda4 * loss_delta_of_delta, "lambda5 * loss_crossentropy",lambda5 * loss_crossentropy)
        #print("loss",loss)
        return lambda1 * loss_mse, lambda2 * loss_cos, lambda3 * loss_delta, lambda4 * loss_delta_of_delta, lambda5 * loss_crossentropy, loss


def delta_list(input: Tensor, labels: Tensor):
    '''
    :param input: (max number of sentences in stories, batch size, dimension of sentence embedding)
    :param labels: (batch size, max number of sentences in stories), masked position == 0 or 1, else == token number of "<SEN>"
    :return: a 1D float tensor of similarity between each sentence and the next with padding sentences excluded
    '''
    sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    input = input.transpose(0, 1).type(torch.DoubleTensor).to(DEVICE)
    deltas = []
    for n_batch, batch in enumerate(input):
        for n_sen in range(batch.size()[0] - 1):
            if (labels[n_batch][n_sen].item() != 0 and labels[n_batch][n_sen].item() != 1) and \
                    (labels[n_batch][n_sen + 1].item() != 0 and labels[n_batch][n_sen + 1].item() != 1):
                delta = sim(batch[n_sen:n_sen + 1], batch[n_sen + 1:n_sen + 2]).item()
                deltas.append(delta)
    return torch.tensor(deltas) if deltas != [] else torch.tensor(
        [0.]).to(DEVICE)  # in case there is only one sentence and the delta list is empty


def delta_of_delta_list(delta_list: Tensor):
    '''
    :param delta_list: a 1D float tensor of similarity between each sentence and the next with padding sentences excluded
    :return: a 1D float tensor of changing rate between each similarity score with padding sentences excluded
    '''
    delta_of_delta_list = []
    for n in range(len(delta_list) - 1):
        if delta_list[n].item() != 0:
            delta_of_delta = ((delta_list[n + 1].item() - delta_list[n].item()) / delta_list[n].item())/100
        else:
            delta_of_delta = ((delta_list[n + 1].item() - delta_list[n].item()) / 1e-6)/100
        delta_of_delta_list.append(delta_of_delta)
    return torch.tensor(delta_of_delta_list) if delta_of_delta_list != [] else torch.tensor(
        [0.]).to(DEVICE)  # in case the delta_of_delta list is empty



class CrossEntropyLoss(nn.Module):
  def __init__(self, tokenizer):
    super(CrossEntropyLoss,self).__init__()
    self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)
    self.vocab_size = len(tokenizer)
  def forward(self, logits, labels):
    loss = self.CrossEntropyLoss(logits.view(-1,self.vocab_size), labels.flatten())
    return loss