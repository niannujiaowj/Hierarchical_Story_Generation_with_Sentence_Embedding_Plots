import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import time
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import h5py

from Scripts.MyModel.transformer import Models
from Scripts.MyModel.transformer import Tgt_Out
from Scripts.MyModel.transformer import SenEmbedding_Loss
from Scripts.MyModel.util import load_checkpoint, EarlyStopping
from Scripts.MyModel.preprocess import *

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device() # google colab tpu
except:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
    return mask.to(DEVICE)



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

    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)



def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    # src (max length of tokens in prompts, batch size), tgt (max number of sentences in stories, batch size)
    for src, tgt in tqdm(train_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        if MODE == "WritingPrompts2SenEmbeddings":
            # logits (max number of sentences in stories, batch size, dimension of sentence embedding)
            start_time = time.time()
            logits, is_pad = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX, TRAIN_SENEMBEDDING_DICT_FILE_PATH)
            #print("logits",logits,logits.size())
            #print("is_pad",is_pad,is_pad.size())
            end_time = time.time()
            print("Calling model",end_time-start_time)

            start_time = time.time()
            optimizer.zero_grad()
            end_time = time.time()
            print("optimizer", end_time - start_time)

            start_time = time.time()
            tgt_out = Tgt_Out(tgt, TRAIN_SENEMBEDDING_DICT).to(DEVICE)
            end_time = time.time()
            print("getting ground truth", end_time - start_time)

            start_time = time.time()
            loss = loss_fn(logits, is_pad, tgt_out, tgt_padding_mask, LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5)
            end_time = time.time()
            print("loss", end_time - start_time)

        else:
            logits = model(src, tgt_input, src_mask, tgt_mask,
                                               src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX,
                                               TRAIN_SENEMBEDDING_DICT_FILE_PATH)
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
    for src, tgt in tqdm(val_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        if MODE =="WritingPrompts2SenEmbeddings":
            logits, is_pad = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX, VALID_SENEMBEDDING_DICT_FILE_PATH)
            tgt_out = Tgt_Out(tgt, VALID_SENEMBEDDING_DICT).to(DEVICE)
            loss = loss_fn(logits, is_pad, tgt_out, tgt_padding_mask, LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5)
        else:
            logits = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask, PAD_IDX, VALID_SENEMBEDDING_DICT_FILE_PATH)
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
    parser.add_argument('--CHECKPOINT_SVAE_PATH', type=str, default="Checkpoints")
    parser.add_argument('--BEST_MODEL_SAVE_PATH',type=str, default="BestModel")
    parser.add_argument('--TRAIN_SENEMBEDDING_DICT', type=str,
                        default="Data/WritingPrompts2SenEmbeddings/train_SenEmbedding_dict.hdf5")
    parser.add_argument('--VALID_SENEMBEDDING_DICT', type=str,
                        default="Data/WritingPrompts2SenEmbeddings/valid_SenEmbedding_dict.hdf5")
    parser.add_argument('--TEST_SENEMBEDDING_DICT', type=str,
                        default="Data/WritingPrompts2SenEmbeddings/test_SenEmbedding_dict.hdf5")
    parser.add_argument('--NUM_ENCODER_LAYERS', type=int, default = 12)
    parser.add_argument('--NUM_DECODER_LAYERS', type=int, default = 12)
    parser.add_argument('--EMB_SIZE', type=int, default = 768)
    parser.add_argument('--FFN_HID_DIM', type=int, default = 768)
    parser.add_argument('--DROPOUT', type=float, default = 0.1)
    parser.add_argument('--NHEAD', type=int, default = 8)
    parser.add_argument('--BATCH_SIZE', type=int, default = 128)
    parser.add_argument('--NUM_EPOCHS', type=int, default = 16)
    parser.add_argument('--RESUME_TRAINING', action='store_true')
    parser.add_argument('--LAMBDA1',type=float, default = 1.0)
    parser.add_argument('--LAMBDA2',type=float, default = 1.0)
    parser.add_argument('--LAMBDA3',type=float, default = 3.)
    parser.add_argument('--LAMBDA4',type=float, default = 2.0)
    parser.add_argument('--LAMBDA5',type=float, default = 0.1)
    parser.add_argument('--PATIENCE', type=int, default=20)
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    MODE = args.MODE
    src_vocab = torch.load(args.SRC_VOCAB)
    tgt_vocab = torch.load(args.TGT_VOCAB)
    train_data = torch.load(args.TRAIN_DATA)
    valid_data = torch.load(args.VALID_DATA)
    test_data = torch.load(args.TEST_DATA)
    TRAIN_SENEMBEDDING_DICT_FILE_PATH = args.TRAIN_SENEMBEDDING_DICT
    VALID_SENEMBEDDING_DICT_FILE_PATH = args.VALID_SENEMBEDDING_DICT
    #TEST_SENEMBEDDING_DICT_FILE_PATH = args.TEST_SENEMBEDDING_DICT
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
    CHECKPOINT_PATH = args.CHECKPOINT_SVAE_PATH
    BEST_MODEL_PATH = args.BEST_MODEL_SAVE_PATH
    LAMBDA1 = args.LAMBDA1
    LAMBDA2 = args.LAMBDA2
    LAMBDA3 = args.LAMBDA3
    LAMBDA4 = args.LAMBDA4
    LAMBDA5 = args.LAMBDA5
    PATIENCE = args.PATIENCE
    TRAIN_SENEMBEDDING_DICT = h5py.File(TRAIN_SENEMBEDDING_DICT_FILE_PATH, "r")
    VALID_SENEMBEDDING_DICT = h5py.File(VALID_SENEMBEDDING_DICT_FILE_PATH, "r")
    #TEST_SENEMBEDDING_DICT = h5py.File(TEST_SENEMBEDDING_DICT_FILE_PATH, "r")

    print("type of dic",type(TRAIN_SENEMBEDDING_DICT))

    '''for file in TRAIN_SENEMBEDDING_DICT.keys():
        print("file",file)
        for i in file:
            print("item in file",i)'''

    UNK_IDX = src_vocab['<unk>'] # 0
    PAD_IDX = src_vocab['<pad>'] # 1
    BOS_IDX = src_vocab['<bos>'] # 2
    EOS_IDX = src_vocab['<eos>'] # 3





    # *_iter is composed of tuples. One tuple one batch. Each batch is a (source, target) tuple pair. The max length
    # varies according to each batch.
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, drop_last=True)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, drop_last=True)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch, drop_last=True)

    start_time = time.time()
    transformer = Models.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                            EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                            FFN_HID_DIM, DROPOUT, NHEAD, MODE, BATCH_SIZE)
    end_time = time.time()
    print("defining the model",end_time-start_time)
    if MODE == "WritingPrompts2SenEmbeddings":
        loss_fn = SenEmbedding_Loss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    start_epoch = 1
    EarlyStopping_counter = 0

    if RESUME_TRAINING:
        transformer, optimizer, start_epoch = load_checkpoint(CHECKPOINT_PATH, transformer, optimizer)
        EarlyStopping_counter = torch.load("{}/{}_EarlyStopping_counter.pt".format(BEST_MODEL_PATH,MODE))

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)


    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)

    for epoch in tqdm(range(start_epoch, NUM_EPOCHS + 1)):
        start_time = time.time()
        print("Processing training data...")
        train_loss = train_epoch(transformer, train_iter, optimizer)
        end_time = time.time()
        print("Processing valid data...")
        val_loss = evaluate(transformer, valid_iter)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
               f"Val loss: {val_loss:.3f}, "
               f"Epoch time = {(end_time - start_time):.3f}s"))

        # save checkpoint for resume training
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, CHECKPOINT_PATH+"/{}.tar".format(MODE))

        # save best model and perform early stopping
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, counter=EarlyStopping_counter, path=BEST_MODEL_PATH)
        early_stopping(val_loss, transformer, MODE)
        if early_stopping.early_stop:
            print("Early stopping")
            break

