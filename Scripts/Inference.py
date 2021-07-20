import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from train import generate_square_subsequent_mask
from Scripts.transformer import Models

def generate_batch(data):
    '''
    args: data: a list of (source, target) tensor pairs
            [(tensor([7,10,8,30,259,18,443]),tensor([4,5,6,..,25])),
            (tensor([]),tensor([26,27,...,83])),
            ...,
            ()]
    output: source_batch: padded tensor (max length of tokens in prompts, number of prompt lines)
        target_batch: padded tensor (max number of sentences in stories, number of stories)
    '''

    source_batch, target_batch= [], []
    for (source_item,target_item) in data:
        source_batch.append(source_item)
        if MODE == "WritingPrompts2SenEmbeddings":
            target_batch.append(torch.cat([torch.tensor([BOS_IDX]), target_item], dim=0))
        else:
            # for the tensor of each story , add <bos> and <eos> token indices at the beginning and the end, then append it to a list
            target_batch.append(torch.cat([torch.tensor([BOS_IDX]), target_item, torch.tensor([EOS_IDX])], dim=0))

    source_batch = pad_sequence(source_batch, padding_value=PAD_IDX)
    target_batch = pad_sequence(target_batch, padding_value=PAD_IDX)
    return source_batch, target_batch

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    '''
    :param model
    :param src (max length of tokens in prompts * batch size, 1)
    :param src_mask (max length of tokens in prompts * batch size, max length of tokens in prompts * batch size) all False
    :param max_len
    :param start_symbol
    :return:
    '''
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    # memory (max length of tokens in prompts * batch size, 1, emb_size)
    memory = model.encode(src, src_mask)
    # ys (1, 1), e.g., tensor([[2]]), filled with <bos> index number
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        #tgt_mask (ys.size(0), ys.size(0)), e.g.,
        #[[False, True, True, ..., True, True, True],
        # [False, False, True, ..., True, True, True],
        # [False, False, False, ..., True, True, True],
        # [...]
        # [False, False, False, ..., False, True, True],
        # [False, False, False, ..., False, False, True]]
        tgt_mask = (generate_square_subsequent_mask(ys.size(0),DEVICE)
                    .type(torch.bool)).to(DEVICE)
        print("ys.size(0)",ys.size(0))
        # out (ys.size(0), 1, emb_size)
        out = model.decode(ys, memory, tgt_mask, PAD_IDX)
        # out (1, ys.size(0), emb_size)
        out = out.transpose(0, 1)
        print("out",out,out.size())
        print("out[:, -1]",out[:, -1],out[:, -1].size())
        # out[:, -1] gets the last embedding
        prob = model.generator(out[:, -1])
        print("prob",prob,prob.size())
        if MODE == "WritingPrompts2SenEmbeddings":
            #sigmoid = nn.Sigmoid()
            #prob = sigmoid(prob)
            if prob > 0.5:
                next_word = torch.tensor([1]) # is pad
            else:
                next_word = torch.tensor([0]) # unpad
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == PAD_IDX:
                break

        else:
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
    return ys


def translate(model, test_iter):
    model.eval()
    # src (max length of tokens in prompts, batch size)
    for src, tgt in test_iter:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # src (max length of tokens in prompts * batch size, 1)
        src = src.view(-1, 1)
        num_tokens = src.size()[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        #return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")



if __name__ == '__main__':
    '''parser = argparse.ArgumentParser()
    parser.add_argument('MODE', type=str, choices=['WritingPrompts2SenEmbeddings', 'Plots2Stories'])
    parser.add_argument('MODEL_PATH', type=str)
    parser.add_argument('SRC_VOCAB', type=str)
    parser.add_argument('TEST_DATA', type=str)
    parser.add_argument('BATCH_SIZE', type=int, default = 128)
    args = parser.parse_args()

    MODE = args.MODE
    MODEL_PATH = args.MODEL_PATH
    src_vocab = torch.load(args.SRC_VOCAB)
    UNK_IDX = src_vocab['<unk>']  # 0
    PAD_IDX = src_vocab['<pad>']  # 1
    BOS_IDX = src_vocab['<bos>']  # 2
    EOS_IDX = src_vocab['<eos>']  # 3
    test_data = torch.load(args.TEST_DATA)
    BATCH_SIZE = args.BATCH_SIZE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')'''


    MODE = "WritingPrompts2SenEmbeddings"
    MODEL_PATH = "Checkpoints/WritingPrompts2SenEmbeddings.tar"
    src_vocab = torch.load("Data/WritingPrompts2SenEmbeddings/src_vocab.pt")
    tgt_vocab = torch.load("Data/WritingPrompts2SenEmbeddings/tgt_vocab.pt")
    UNK_IDX = 0
    PAD_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3
    test_data = torch.load("Data/WritingPrompts2SenEmbeddings/train_data.pt")
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    EMB_SIZE = 768
    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab) - 1
    FFN_HID_DIM = 768
    DROPOUT = 0.1
    NHEAD = 8
    BATCH_SIZE = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if MODE == 'WritingPrompts2SenEmbeddings':
        state_dict = torch.load(MODEL_PATH)["model_state_dict"]
        model = Models.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                     EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                     FFN_HID_DIM, DROPOUT, NHEAD, MODE, BATCH_SIZE)
        model.load_state_dict(state_dict)
        test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch, drop_last=True)
        translate(model,test_iter)
    else:
        pass