import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import time
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from Scripts.transformer import Models
from Scripts.transformer.util import Tgt_Out
from Scripts.transformer.Modules import SenEmbedding_Loss
from Scripts.util import load_checkpoint
from Scripts.preprocess import *





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



def generate_square_subsequent_mask(sz,device):
    '''
    args: sz: max number of sentences in stories
    output: mask: tensor (max number of sentences in stories, max number of sentences in stories)
    [[0., -inf, -inf, ..., -inf, -inf, -inf],
    [0., 0. ,-inf, ..., -inf, -inf, -inf],
    [0., 0., 0., ..., -inf, -inf, -inf],
    [...]
    [0., 0., 0., ..., 0., 0., -inf],
    [0., 0., 0., ..., 0., 0., 0.]]
    '''
    # torch.triu(input, diagonal=0, *, out=None) returns the upper triangular part of a matrix (2-D tensor) or batch
    # of matrices input, the other elements of the result tensor out are set to 0.
    # Transpose returns the lower triangular part of a matrix == True, else == False
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # mask.float(), True --> 1., Flase --> 0.
    # masked_fill(mask == 0, float('-inf')), 0. --> '-inf'
    # masked_fill(mask == 1, float(0.0)), 1. --> 0.
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



def create_mask(src, tgt):
    '''
    args: src, tgt: one batch data in *_iter
    output: src_mask: tensor (max length of tokens in prompts, max length of tokens in prompts), all False
        tgt_mask: tensor (max number of sentences in stories, max number of sentences in stories)
        [[0., -inf, -inf, ..., -inf, -inf, -inf],
        [0., 0. ,-inf, ..., -inf, -inf, -inf],
        [0., 0., 0., ..., -inf, -inf, -inf],
        [...]
        [0., 0., 0., ..., 0., 0., -inf],
        [0., 0., 0., ..., 0., 0., 0.]]
        src_padding_mask: tensor (batch size, max length of tokens in prompts),
                        masked position == True, else == False
        tgt_padding_mask: tensor (batch size, max number of sentences in stories),
                        masked position == True, else == False
    '''
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,DEVICE)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    # src (max length of tokens in prompts, batch size), tgt (max number of sentences in stories, batch size)
    for idx, (src, tgt) in enumerate(train_iter):
        print("Train: batch {} out of {}".format(idx+1,len(train_iter)))
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        if MODE == "WritingPrompts2SenEmbeddings":
            # logits (max number of sentences in stories, batch size, dimension of sentence embedding)
            logits, is_pad = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX, TRAIN_SENEMBEDDING_DICT)
            #print("logits",logits,logits.size())
            #print("is_pad",is_pad,is_pad.size())
            optimizer.zero_grad()
            tgt_out = Tgt_Out(tgt, "Data/WritingPrompts2SenEmbeddings/train_SenEmbedding_dict.pt")
            loss = loss_fn(logits, is_pad, tgt_out, tgt_padding_mask, LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5)
        else:
            logits = model(src, tgt_input, src_mask, tgt_mask,
                                               src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX,
                                               TRAIN_SENEMBEDDING_DICT)
            # exlude the <bos> token in each story
            optimizer.zero_grad()
            tgt_out = tgt[1:,:]
            # tgt_out.reshape(-1) flattens tgt_out
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)



def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in enumerate(val_iter):
        print("Valid: batch {} out of {}".format(idx+1, len(val_iter)))
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        if MODE =="WritingPrompts2SenEmbeddings":
            logits, is_pad = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX, VALID_SENEMBEDDING_DICT)
            tgt_out = Tgt_Out(tgt, "Data/WritingPrompts2SenEmbeddings/valid_SenEmbedding_dict.pt")
            loss = loss_fn(logits, is_pad, tgt_out, tgt_padding_mask, LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5)
        else:
            logits = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX, VALID_SENEMBEDDING_DICT)
            tgt_out = tgt[1:,:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODE', type=str, choices=['WritingPrompts2SenEmbeddings', 'Plots2Stories'])
    parser.add_argument('SRC_VOCAB', type=str)
    parser.add_argument('TGT_VOCAB', type=str)
    parser.add_argument('TRAIN_DATA', type=str)
    parser.add_argument('VALID_DATA', type=str)
    parser.add_argument('TEST_DATA', type=str)
    parser.add_argument('--SAVE_PATH', type=str, default="Checkpoints")
    parser.add_argument('--TRAIN_SENEMBEDDING_DICT', type=str,
                        default="Data/WritingPrompts2SenEmbeddings/train_SenEmbedding_dict.pt")
    parser.add_argument('--VALID_SENEMBEDDING_DICT', type=str,
                        default="Data/WritingPrompts2SenEmbeddings/valid_SenEmbedding_dict.pt")
    parser.add_argument('--TEST_SENEMBEDDING_DICT', type=str,
                        default="Data/WritingPrompts2SenEmbeddings/test_SenEmbedding_dict.pt")
    parser.add_argument('--NUM_ENCODER_LAYERS', type=int, default = 12)
    parser.add_argument('--NUM_DECODER_LAYERS', type=int, default = 12)
    parser.add_argument('--EMB_SIZE', type=int, default = 768)
    parser.add_argument('--FFN_HID_DIM', type=int, default = 768)
    parser.add_argument('--DROPOUT', type=float, default = 0.1)
    parser.add_argument('--NHEAD', type=int, default = 8)
    parser.add_argument('--BATCH_SIZE', type=int, default = 128)
    parser.add_argument('--NUM_EPOCHS', type=int, default = 16)
    parser.add_argument('--RESUME_TRAINING', action='store_true')
    parser.add_argument('--CHECKPOINT_PATH',type=str)
    parser.add_argument('--LAMBDA1',type=float, default = 1.0)
    parser.add_argument('--LAMBDA2',type=float, default = 1.0)
    parser.add_argument('--LAMBDA3',type=float, default = 5.)
    parser.add_argument('--LAMBDA4',type=float, default = 1.0)
    parser.add_argument('--LAMBDA5',type=float, default = 0.1)
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    MODE = args.MODE
    src_vocab = torch.load(args.SRC_VOCAB)
    tgt_vocab = torch.load(args.TGT_VOCAB)
    train_data = torch.load(args.TRAIN_DATA)
    #valid_data = torch.load(args.VALID_DATA)
    #test_data = torch.load(args.TEST_DATA)
    SAVE_PATH = args.SAVE_PATH
    TRAIN_SENEMBEDDING_DICT = torch.load(args.TRAIN_SENEMBEDDING_DICT)
    #VALID_SENEMBEDDING_DICT = torch.load(args.VALID_SENEMBEDDING_DICT)
    #TEST_SENEMBEDDING_DICT = torch.load(args.TEST_SENEMBEDDING_DICT)
    NUM_ENCODER_LAYERS = args.NUM_ENCODER_LAYERS
    NUM_DECODER_LAYERS = args.NUM_DECODER_LAYERS
    EMB_SIZE = args.EMB_SIZE
    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab) - 1 # simplify to binary classification problem, the num of last layer' neural network is 1
    FFN_HID_DIM = args.FFN_HID_DIM
    DROPOUT = args.DROPOUT
    NHEAD = args.NHEAD
    BATCH_SIZE = args.BATCH_SIZE
    NUM_EPOCHS = args.NUM_EPOCHS
    RESUME_TRAINING = args.RESUME_TRAINING
    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    LAMBDA1 = args.LAMBDA1
    LAMBDA2 = args.LAMBDA2
    LAMBDA3 = args.LAMBDA3
    LAMBDA4 = args.LAMBDA4
    LAMBDA5 = args.LAMBDA5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    UNK_IDX = src_vocab['<unk>'] # 0
    PAD_IDX = src_vocab['<pad>'] # 1
    BOS_IDX = src_vocab['<bos>'] # 2
    EOS_IDX = src_vocab['<eos>'] # 3






    # *_iter is composed of tuples. One tuple one batch. Each batch is a (source, target) tuple pair. The max length
    # varies according to each batch.
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, drop_last=True)
    #valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, drop_last=True)
    #test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=Flase, collate_fn=generate_batch, drop_last=True)



    transformer = Models.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                     EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                     FFN_HID_DIM, DROPOUT, NHEAD, MODE, BATCH_SIZE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)


    if MODE == "WritingPrompts2SenEmbeddings":
        loss_fn = SenEmbedding_Loss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    valid_loss_min = np.Inf
    if RESUME_TRAINING:
        model, optimizer, start_epoch, valid_loss_min = load_checkpoint(args.CHECKPOINT_PATH, transformer, optimizer)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        start_time = time.time()
        train_loss = train_epoch(transformer, train_iter, optimizer)
        end_time = time.time()
        #val_loss = evaluate(transformer, valid_iter)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
               #f"Val loss: {val_loss:.3f}, "
               f"Epoch time = {(end_time - start_time):.3f}s"))

        # save checkpoint for resume training
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            #'val_loss': val_loss
        }, SAVE_PATH+"/{}.tar".format(MODE))

        # save the best model
        '''if val_loss <= valid_loss_min:
            valid_loss_min = val_loss
            torch.save(transformer.state_dict(),"BestModel/{}.pt".format(MODE))'''
