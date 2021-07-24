from collections import Counter
import io
import os
import argparse
import gc
from tqdm import tqdm
import numpy as np
import torch
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer




def build_vocab(filepath, tokenizer):
    '''
    args: filepath: file path
        tokenizer: spacy tokenizer, e.g. en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    output: PyTorch vocabulary with indices for tokens
    '''

    # build a {token: count} dictionary
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        # tokenize each line of the file
        for string_ in tqdm(f):
            counter.update(tokenizer(string_))
    # build a PyTorch vocabulary with indices for tokens
    specials = {'<unk>':0, '<pad>':1, '<bos>':2, '<eos>':3}
    vocabulary = vocab(counter)
    for k,v in specials.items():
        vocabulary.insert_token(k,v)
    vocabulary.set_default_index(0)
    return vocabulary





def data_process(filepath_src:str, filepath_tgt: str, tokenizer, vocab_src, vocab_tgt, mode: str,
                 file_path_SenEmbedding_dict: str = None, filepath_src_senindices: str = None):
    '''
    :param filepath_src: WritingPrompts2SenEmbeddings: writing prompts
                        Plots2Stories: plots
    :param filepath_tgt: WritingPrompts2SenEmbeddings: arrays of sentence embeddings (.npy)
                        Plots2Stories:stories
    :param tokenizer: tokenizer
    :param vocab_src: vocabulary of source training file
    :param vocab_tgt: vocabulary of target training file
    :param mode: WritingPrompts2SenEmbeddings or Plots2Stories
    :param file_path_SenEmbedding_dict: WritingPrompts2SenEmbeddings: dictionary of sentence embeddings (.hdf5)
    :param filepath_src_senindices: Plots2Stories: list of indices of sentence embeddings (.pt)
    :return: WritingPrompts2SenEmbeddings:
                data: a list of (source, target) tensor pairs
                    [(tensor([7,10,8,30,259,18,...,443]),tensor([100000,100001,100002,..,100025])),
                    (tensor([7,34,62,435,76,24,...,654]),tensor([100026,100027,100028,...,100054])),
                    ...,
                    ()]
                SenIndices: the list of sentence indices.
                    [[100000,100001,100002,..,100025],[100026,100027,100028,...,100054],...,[]]
                SenEmbedding_dict: the dictionary of sentence indices and sentence embeddings. The dimension of each
                    sentence embedding is 768
                    {100000:[0.0952,...,0.1465],100001:[-0.01092,...,0.0753],...,100025:[0.1384,...,-0.0096]}
            Plots2Stories:
                data: a list of (source, target) tensor pairs. The source tensor is the combination of indices of
                     plots tokens and sentence embeddings
                    [(tensor([7,10,8,30,259,18,...,443,100000,100001,100002,..,100025]),tensor([97,64,753,..,23])),
                    (tensor([7,34,62,435,76,24,...,654,100026,100027,100028,...,100054]),tensor([545,65,...,79])),
                    ...,
                    ()]
    '''
    import h5py
    import re
    data = []
    if mode == "WritingPrompts2SenEmbeddings":
        # iterate each item (each line of the source file, each story's sentence embeddings of the target file)
        raw_source_iter = iter(io.open(filepath_src, encoding="utf8"))
        target = np.load(filepath_tgt, allow_pickle=True)
        n_sen = 10000000  # the vocabulary size of plots is 67932??

        if re.match(".*npz$", filepath_tgt):
            n = 0
            for file in target.files:
                n += len(target[file])
            SenIndices = [[] for _ in range(n)]
            with h5py.File(file_path_SenEmbedding_dict, "w") as f:
                print("Saving sentence embedding dictionary...")
                for file in tqdm(target.files):
                    for n_story, story in tqdm(enumerate(target[file])):
                        for sen in story:
                            f.create_dataset(str(n_sen), data = sen)
                            SenIndices[n_story].append(n_sen)
                            n_sen += 1
                print("Done!")
        else:
            SenIndices = [[] for _ in range(len(target))]
            with h5py.File(file_path_SenEmbedding_dict,"w") as f:
                print("Saving sentence embedding dictionary...")
                for n_story,story in tqdm(enumerate(target)):
                    for sen in story:
                        f.create_dataset(str(n_sen), data = sen)
                        SenIndices[n_story].append(n_sen)
                        n_sen += 1
                print("Done!")

        raw_target_iter = iter(SenIndices)

        # combine source and target data
        for (raw_source, raw_target) in tqdm(zip(raw_source_iter, raw_target_iter)):
            # get the token id from the vocabulary
            source_tensor_ = torch.tensor([vocab_src[token] for token in tokenizer(raw_source)], dtype=torch.long)
            target_tensor_ = torch.tensor(raw_target)
            data.append((source_tensor_, target_tensor_))
        return data, SenIndices
    else:
        raw_source_plot_iter = iter(io.open(filepath_src, encoding="utf8"))
        raw_source_sen_iter = iter(torch.load(filepath_src_senindices))
        raw_target_iter = iter(io.open(filepath_tgt, encoding="utf8"))

        for (raw_source_plot, raw_source_sen, raw_target) in tqdm(zip(raw_source_plot_iter, raw_source_sen_iter, raw_target_iter)):
            source_tensor_plot = torch.tensor([vocab_src[token] for token in tokenizer(raw_source_plot)], dtype=torch.long)
            source_tensor_sen = torch.tensor(raw_source_sen, dtype=torch.long)
            source_tensor_ = torch.cat((source_tensor_plot, source_tensor_sen), dim = 0)
            target_tensor_ = torch.tensor([vocab_tgt[token] for token in tokenizer(raw_target)], dtype=torch.long)

            data.append((source_tensor_,target_tensor_))
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('MODE', type=str, choices=['WritingPrompts2SenEmbeddings', 'Plots2Stories'])
    parser.add_argument('DATASET', type=str, choices=['TRAIN', 'VALID', "TEST", "ALL"])
    parser.add_argument('TRAIN_SRC',type=str)
    parser.add_argument('--TRAIN_SRC_SENINDICES', type=str)
    parser.add_argument('TRAIN_TGT',type=str)
    parser.add_argument('VALID_SRC',type=str)
    parser.add_argument('--VALID_SRC_SENINDICES', type=str)
    parser.add_argument('VALID_TGT',type=str)
    parser.add_argument('TEST_SRC',type=str)
    parser.add_argument('--TEST_SRC_SENINDICES', type=str)
    parser.add_argument('TEST_TGT',type=str)
    parser.add_argument('--SAVE_PATH', type=str)
    args = parser.parse_args()

    DATASET = args.DATASET
    en_tokenizer = get_tokenizer("basic_english")
    if args.MODE == "WritingPrompts2SenEmbeddings":
        if args.SAVE_PATH is None:
            SAVE_PATH = "Data/WritingPrompts2SenEmbeddings"
        else:
            SAVE_PATH = args.SAVE_PATH
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        print("Building source vocabulary...")
        src_vocab = build_vocab(args.TRAIN_SRC, en_tokenizer)
        print("Saving source vocabulary...")
        torch.save(src_vocab, SAVE_PATH + "/src_vocab.pt")
        print("Done!")


        print("Building target vocabulary...")
        counter = Counter()
        specials = {'<unpad>':0, '<pad>':1}
        tgt_vocab = vocab(counter)
        for k,v in specials.items():
            tgt_vocab.insert_token(k,v)
        print("Saving target vocabulary...")
        torch.save(tgt_vocab, SAVE_PATH + "/tgt_vocab.pt")
        print("Done!")
        del tgt_vocab
        gc.collect()

        if DATASET == "TRAIN" or DATASET == "ALL":
            print("Processing training data...")
            train_data, train_SenIndices = data_process(
                args.TRAIN_SRC, args.TRAIN_TGT, en_tokenizer, src_vocab, src_vocab, args.MODE,
                os.path.join(SAVE_PATH + "/train_SenEmbedding_dict.hdf5"))
            print("Saving training data...")
            torch.save(train_data, SAVE_PATH + "/train_data.pt")
            del train_data
            gc.collect()
            print("Saving training sentence indices...")
            torch.save(train_SenIndices, SAVE_PATH + "/train_SenIndices.pt")
            del train_SenIndices
            gc.collect()
            print("Done!")

        if DATASET == "VALID" or DATASET == "ALL":
            print("Processing valid data...")
            valid_data, valid_SenIndices = data_process(
                args.VALID_SRC, args.VALID_TGT, en_tokenizer, src_vocab, src_vocab, args.MODE,
                os.path.join(SAVE_PATH + "/valid_SenEmbedding_dict.hdf5"))
            print("Saving valid data...")
            torch.save(valid_data, SAVE_PATH + "/valid_data.pt")
            del valid_data
            gc.collect()
            print("Saving valid sentence indices...")
            torch.save(valid_SenIndices, SAVE_PATH + "/valid_SenIndices.pt")
            del valid_SenIndices
            gc.collect()
            print("Done!")

        if DATASET == "TEST" or DATASET == "ALL":
            print("Processing test data...")
            test_data, test_SenIndices = data_process(
                args.TEST_SRC, args.TEST_TGT, en_tokenizer, src_vocab, src_vocab, args.MODE,
                os.path.join(SAVE_PATH + "/test_SenEmbedding_dict.hdf5"))
            print("Saving test data...")
            torch.save(test_data, SAVE_PATH + "/test_data.pt")
            del test_data
            gc.collect()
            print("Saving test sentence indices...")
            torch.save(test_SenIndices, SAVE_PATH + "/test_SenIndices.pt")
            del test_SenIndices
            gc.collect()
            print("Done!")



    else:
        if args.SAVE_PATH is None:
            SAVE_PATH = "Data/Plots2Stories"
        else:
            SAVE_PATH = args.SAVE_PATH
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)


        print("Building source vocabulary...")
        src_vocab = build_vocab(args.TRAIN_SRC, en_tokenizer)
        print("Saving source vocabulary...")
        torch.save(src_vocab, SAVE_PATH + "/src_vocab.pt")
        print("Done!")


        print("Building target vocabulary...")
        tgt_vocab = build_vocab(args.TRAIN_TGT, en_tokenizer)
        print("Saving target vocabulary...")
        torch.save(tgt_vocab, SAVE_PATH + "/tgt_vocab.pt")
        print("Done!")

        if DATASET == "TRAIN" or DATASET == "ALL":
            print("Processing training data...")
            train_data = data_process(args.TRAIN_SRC, args.TRAIN_TGT, en_tokenizer, src_vocab, tgt_vocab,
                                      args.MODE, filepath_src_senindices = args.TRAIN_SRC_SENINDICES)
            print("Saving training data...")
            torch.save(train_data, SAVE_PATH + "/train_data.pt")
            print("Done!")
            del train_data
            gc.collect()

        if DATASET == "VALID" or DATASET == "ALL":
            print("Processing valid data...")
            valid_data = data_process(args.VALID_SRC, args.VALID_TGT, en_tokenizer, src_vocab, tgt_vocab,
                                      args.MODE, filepath_src_senindices = args.VALID_SRC_SENINDICES)
            print("Saving valid data...")
            torch.save(valid_data, SAVE_PATH + "/valid_data.pt")
            print("Done!")
            del valid_data
            gc.collect()

        if DATASET == "TEST" or DATASET == "ALL":
            print("Processing test data...")
            test_data = data_process(args.TEST_SRC, args.TEST_TGT, en_tokenizer, src_vocab, tgt_vocab,
                                     args.MODE, filepath_src_senindices = args.TEST_SRC_SENINDICES)
            print("Saving test data...")
            torch.save(test_data, SAVE_PATH + "/test_data.pt")
            print("Done!")
            del test_data
            gc.collect()










