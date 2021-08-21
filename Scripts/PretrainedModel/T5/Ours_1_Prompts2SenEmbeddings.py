import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import re
import argparse
import gc
from tqdm import tqdm
from datetime import date
import h5py
#import glob
#import json
#import time
#import logging
#import random
#from itertools import chain
#from string import  punctuation

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
#import nlp
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AdamW, T5TokenizerFast, T5ForConditionalGeneration
#from transformers import get_linear_schedule_with_warmup
from utils import SenEmbedding_Loss


def prepare_data(input_path: Path, target_path: Path):
    with open(input_path) as input_file:
        inputs = input_file.readlines()
    print("Start loading data...")
    targets = np.load(target_path, allow_pickle=True)
    data_rows = []
    start_story = 0
    print("Start genrating data...")
    if re.match(".*npz$", target_path):
        for file in tqdm(targets.files):
            end_story = len(targets[file]) + start_story
            for line_in, story_senembeddings in zip(inputs[start_story:end_story], targets[file]):
                data_rows.append({"source": line_in, "target": "<SEN> " * len(story_senembeddings),
                                  "targets_senembeddings": story_senembeddings})
            start_story += len(file)
    else:
        for line_in, story_senembeddings in zip(inputs, targets):
            data_rows.append({"source": line_in, "target": "<SEN> " * len(story_senembeddings),
                              "targets_senembeddings": story_senembeddings})

    del targets
    del inputs
    gc.collect()

    return pd.DataFrame(data_rows)





class StorytellingDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5TokenizerFast,
            source_max_token_len: int = 1000,
            target_max_token_len: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        sources_encoding = self.tokenizer(
            data_row["source"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        targets_encoding = self.tokenizer(
            data_row["target"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = targets_encoding["input_ids"]
        labels[labels == 0] = -100 # cannot be used in T5Model


        targets_senembeddings = torch.tensor(data_row["targets_senembeddings"])
        unique_counts = labels.flatten().unique(return_counts=True)
        unique = unique_counts[0]
        counts = unique_counts[1]
        location_sen = (unique == 32100).nonzero(as_tuple=True)[0]

        len_targets_senembeddings = targets_senembeddings.size()[0]
        len_labels = labels.flatten().size()[0]
        len_labels_valid = counts[location_sen]

        eos_embedding = torch.randn(1, targets_senembeddings.size()[1])

        if len_targets_senembeddings < len_labels_valid:
            targets_senembeddings = torch.cat([targets_senembeddings,
                                               eos_embedding,
                                               torch.zeros([len_labels - len_targets_senembeddings - 1,targets_senembeddings.size()[1]])])
        else:
            targets_senembeddings = torch.cat([targets_senembeddings[:len_labels_valid],
                                               eos_embedding,
                                               torch.zeros(len_labels - len_labels_valid - 1, targets_senembeddings.size()[1])])


        return dict(
            source=data_row["source"],
            target=data_row["target"],
            targets_senembeddings=targets_senembeddings,
            input_ids=sources_encoding["input_ids"].flatten(),
            attention_mask=sources_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )




class StorytellingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_DataFrame: pd.DataFrame,
            valid_DataFrame: pd.DataFrame,
            test_DataFrame: pd.DataFrame,
            tokenizer: T5TokenizerFast,
            batch_size: int = 128,
            source_max_token_len: int = 1000,
            target_max_token_len: int = 1000,
            use_tpu=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_DataFrame = train_DataFrame
        self.valid_DataFrame = valid_DataFrame
        self.test_DataFrame = test_DataFrame
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.use_tpu = use_tpu
        self.sampler_train = None
        self.sampler_valid = None
        self.sampler_test = None

    def setup(self):
        self.train_dataset = StorytellingDataset(self.train_DataFrame, self.tokenizer, self.source_max_token_len,
                                                 self.target_max_token_len)
        self.valid_dataset = StorytellingDataset(self.valid_DataFrame, self.tokenizer, self.source_max_token_len,
                                                 self.target_max_token_len)
        self.test_dataset = StorytellingDataset(self.test_DataFrame, self.tokenizer, self.source_max_token_len,
                                                self.target_max_token_len)
        if self.use_tpu:
            import torch_xla.core.xla_model as xm
            self.sampler_train = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                                 num_replicas=xm.xrt_world_size(),
                                                                                 rank=xm.get_ordinal(), shuffle=True)
            self.sampler_valid = torch.utils.data.distributed.DistributedSampler(self.valid_dataset,
                                                                                 num_replicas=xm.xrt_world_size(),
                                                                                 rank=xm.get_ordinal(), shuffle=False)
            self.sampler_test = torch.utils.data.distributed.DistributedSampler(self.test_dataset,
                                                                                num_replicas=xm.xrt_world_size(),
                                                                                rank=xm.get_ordinal(), shuffle=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler=self.sampler_train, batch_size=self.batch_size, num_workers=40,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, sampler=self.sampler_valid, batch_size=self.batch_size, num_workers=40,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, sampler=self.sampler_test, batch_size=self.batch_size, num_workers=40,
                          shuffle=False)


# T5ForConditionalGeneration
class MyModel(nn.Module):
    def __init__(self, model_name, tokenizer, lambda1: float = 1.0, lambda2: float = 1.0, lambda3: float = 1.0,
                 lambda4: float = 1.0, lambda5: float = 1.0):
        super(MyModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True,
                                                                output_hidden_states=True).to(self.device)
        # self.model = T5Model.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        # self.apply(self.init_T5_weights)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5

        self.decoder_embed_layer = nn.Sequential(
            nn.Linear(768, 512, bias=True).to(self.device)
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.Linear(512, 512, bias=False),
            nn.Linear(512, 768, bias=False)
        ).to(self.device)

        # self.vocab_layer = nn.Sequential(
        #    nn.Linear(512, 3, bias=False)
        # ).to(self.device)

    def forward(self, input_ids, attention_mask, labels=None, labels_senembeddings=None):
        '''
        input_ids (batch size, max_src_length)
        attention_mask (batch size, max_src_length)
        labels (batch size, max_tgt_length)
        labels_senembeddings (batch size, max_tgt_length, 768)
        '''
        decoder_inputs_embeds = self.decoder_embed_layer(labels_senembeddings)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             decoder_inputs_embeds=decoder_inputs_embeds)
        last_hidden_states = outputs.decoder_hidden_states[-1]
        outputs_senemb = self.fully_connected_layer(
            last_hidden_states)  # (batch size, max numer of sentences in stories, hidden dimension)
        # logits = self.vocab_layer(last_hidden_states) # (batch size, max numer of sentences in stories, 3)

        logits = outputs.logits  # (batch size, max numer of sentences in stories, vocab size)
        # print("outputs.logits",outputs.logits,outputs.logits.size())
        # print("labels",labels,labels.size()) # (batch size, max numer of sentences in stories)

        loss_fn = SenEmbedding_Loss()
        loss1, loss2, loss3, loss4, loss5, loss = loss_fn(outputs_senemb=outputs_senemb.transpose(0, 1), logits=logits,
                                                          labels_senembeddings=labels_senembeddings.transpose(0, 1),
                                                          labels=labels,
                                                          lambda1=self.lambda1, lambda2=self.lambda2,
                                                          lambda3=self.lambda3,
                                                          lambda4=self.lambda4, lambda5=self.lambda5)

        return loss1, loss2, loss3, loss4, loss5, loss

    def generate(self, input_ids, attention_mask,
                 min_length=None, max_length=50, num_beams=1, repetition_penalty=1.0,
                 length_penalty=1.0, early_stopping=False, use_cache=True):
        outputs = self.model.generate(input_ids=input_ids.to(self.device),
                                      attention_mask=attention_mask.to(self.device),
                                      return_dict_in_generate=True, output_hidden_states=True, min_length=min_length,
                                      max_length=max_length, num_beams=num_beams, repetition_penalty=repetition_penalty,
                                      length_penalty=length_penalty, early_stopping=early_stopping, use_cache=use_cache)

        # (batch size, max length, 512)
        last_hidden_states = torch.cat(
            [outputs.decoder_hidden_states[n][-1] for n in range(len(outputs.decoder_hidden_states))], dim=1).to(
            self.device)
        outputs_senemb = self.fully_connected_layer(last_hidden_states).to(self.device)

        return outputs.sequences, outputs_senemb


class StorytellingModel(pl.LightningModule):
    def __init__(self, model_name, tokenizer, lambda1: float = 1.0, lambda2: float = 1.0, lambda3: float = 3.,
                 lambda4: float = 2.0, lambda5: float = 0.1):
        super().__init__()
        # self.model = MyModel.from_pretrained(model_name, tokenizer = tokenizer)
        self.model = MyModel(model_name, tokenizer, lambda1, lambda2, lambda3, lambda4, lambda5)
        for name, parameter in self.model.named_parameters():
            if re.match(".*shared.*", name) or re.match(".*encoder.*", name):
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, labels_senembeddings=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                             labels_senembeddings=labels_senembeddings)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_senembeddings = batch["targets_senembeddings"]
        _, _, _, _, _, loss = self(input_ids, attention_mask, labels, labels_senembeddings)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_senembeddings = batch["targets_senembeddings"]
        loss1, loss2, loss3, loss4, loss5, loss = self(input_ids, attention_mask, labels, labels_senembeddings)
        print("loss1:", loss1, "loss2:", loss2, "loss3:", loss3, "loss4:", loss4, "loss5:", loss5, "loss:", loss)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss1, loss2, loss3, loss4, loss5, loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels_senembeddings = batch["targets_senembeddings"]
        _, _, _, _, _, loss = self(input_ids, attention_mask, labels, labels_senembeddings)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


class Generate_Story:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def generate_story(self, dataloader, origindata_save_path, senembeds_save_path):
        sen_embeds = []
        n_story = 0
        for batch in tqdm(dataloader):
            outputs_sequences, outputs_senemb = self.model.model.generate(input_ids=batch["input_ids"],
                                                                             attention_mask=batch["attention_mask"],
                                                                             min_length=30,
                                                                             max_length=50,
                                                                             num_beams=1,
                                                                             repetition_penalty=1.0,
                                                                             length_penalty=1.0,
                                                                             early_stopping=True,
                                                                             use_cache=True)

            max_length = outputs_sequences.size()[1] - 1

            for story_seq, story_sen in zip(outputs_sequences, outputs_senemb):
                try:
                    sen_length = (story_seq == 1).nonzero(as_tuple=True)[0].item() - 1
                except:
                    sen_length = max_length
                story_sen = story_sen[:sen_length, :].tolist()
                sen_embeds.append(story_sen)
                with h5py.File(origindata_save_path, "a") as f1:
                    f1.create_dataset("{}/sequences".format(str(n_story)), data=story_seq.cpu())
                    f1.create_dataset("{}/senembedings".format(str(n_story)), data=story_sen)
                n_story += 1
            del outputs_sequences
            del outputs_senemb
            gc.collect()

        print("Saving numpy data...")
        np.save(senembeds_save_path, sen_embeds)
        print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('MODE', type=str, choices=['Train', 'Train_Resume', 'Inference'])
    parser.add_argument('--TRAIN_SRC', type=str, default="Dataset/WritingPrompts/train.wp_source")
    parser.add_argument('--TRAIN_TGT', type=str, default="Dataset/SenEmbedding/train.npz")
    parser.add_argument('--VALID_SRC', type=str, default="Dataset/WritingPrompts/valid.wp_source")
    parser.add_argument('--VALID_TGT', type=str, default="Dataset/SenEmbedding/valid.npy")
    parser.add_argument('--TEST_SRC', type=str, default="Dataset/WritingPrompts/test.wp_source")
    parser.add_argument('--TEST_TGT', type=str, default="Dataset/SenEmbedding/test.npy")
    parser.add_argument('--TRAIN_RESUME_CHECKPOINT_PATH', type=str)
    parser.add_argument('--INFERENCE_BEST_CHECKPOINT_PATH', type=str)
    parser.add_argument('--MODEL_NAME', type=str, default="t5-small")
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--N_EPOCHS', type=int, default=20)
    parser.add_argument('--SOURCE_MAX_TOKEN_LEN', type=int, default=100)
    parser.add_argument('--TARGET_MAX_TOKEN_LEN', type=int, default=50)
    parser.add_argument('--NUM_GPU', type=int, default=1)
    parser.add_argument('--ORIGINDATA_SAVE_PATH', type=str, default="Inference/SenEmbeddings/")
    parser.add_argument('--SENEMBEDS_SAVE_PATH', type=str, default="Inference/SenEmbeddings/")
    parser.add_argument('--CHECKPOINTS_SAVE_PATH', type=str, default="Checkpoints/")
    parser.add_argument('--lambda1', type=float, default=3.0)
    parser.add_argument('--lambda2', type=float, default=1.5)
    parser.add_argument('--lambda3', type=float, default=1.0)
    parser.add_argument('--lambda4', type=float, default=0.000002)
    parser.add_argument('--lambda5', type=float, default=1.0)
    args = parser.parse_args()

    MODE = args.MODE
    MODEL_NAME = args.MODEL_NAME
    TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_NAME)
    TOKENIZER.add_tokens("<SEN>")
    TRAIN_SRC = args.TRAIN_SRC
    TRAIN_TGT = args.TRAIN_TGT
    VALID_SRC = args.VALID_SRC
    VALID_TGT = args.VALID_TGT
    TEST_SRC = args.TEST_SRC
    TEST_TGT = args.TEST_TGT
    TRAIN_RESUME_CHECKPOINT_PATH = args.TRAIN_RESUME_CHECKPOINT_PATH
    INFERENCE_BEST_CHECKPOINT_PATH = args.INFERENCE_BEST_CHECKPOINT_PATH
    BATCH_SIZE = args.BATCH_SIZE
    N_EPOCHS = args.N_EPOCHS
    SOURCE_MAX_TOKEN_LEN = args.SOURCE_MAX_TOKEN_LEN
    TARGET_MAX_TOKEN_LEN = args.TARGET_MAX_TOKEN_LEN
    NUM_GPU = args.NUM_GPU
    ORIGINDATA_SAVE_PATH = args.ORIGINDATA_SAVE_PATH
    SENEMBEDS_SAVE_PATH = args.SENEMBEDS_SAVE_PATH
    CHECKPOINTS_SAVE_PATH = args.CHECKPOINTS_SAVE_PATH
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    lambda3 = args.lambda3
    lambda4 = args.lambda4
    lambda5 = args.lambda5



    if MODE == 'Train' or "Train_Resume":
        train_DataFrame = prepare_data(TRAIN_SRC, TRAIN_TGT)
        valid_DataFrame = prepare_data(VALID_SRC, VALID_TGT)
        test_DataFrame = prepare_data(VALID_SRC, VALID_TGT)
    elif MODE == 'Inference':
        train_DataFrame = prepare_data(TEST_SRC, TEST_TGT)
        valid_DataFrame = prepare_data(TEST_SRC, TEST_TGT)
        test_DataFrame = prepare_data(TEST_SRC, TEST_TGT)


    data_module = StorytellingDataModule(train_DataFrame, valid_DataFrame, test_DataFrame, TOKENIZER,
                                         batch_size=BATCH_SIZE, use_tpu=False,
                                         source_max_token_len=SOURCE_MAX_TOKEN_LEN,
                                         target_max_token_len=TARGET_MAX_TOKEN_LEN)
    data_module.setup()

    if MODE == 'Train':
        model = StorytellingModel(MODEL_NAME, TOKENIZER , lambda1 = lambda1, lambda2 = lambda2, lambda3 = lambda3,
                                  lambda4 = lambda4, lambda5 = lambda5)
        checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                              filename = 'Ours_1_best_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                              save_top_k = 1, verbose = True, monitor = "val_loss", mode = "min")
        trainer = pl.Trainer(callbacks = checkpoint_callback, max_epochs=N_EPOCHS,
                             gpus=NUM_GPU, progress_bar_refresh_rate=30)
        trainer.fit(model,data_module)
    elif MODE == 'Train_Resume':
        model = StorytellingModel(MODEL_NAME, TOKENIZER, lambda1 = lambda1, lambda2 = lambda2, lambda3 = lambda3,
                                  lambda4 = lambda4, lambda5 = lambda5)
        checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                              filename = 'Ours_1_best_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                              save_top_k = 1, verbose = True, monitor = "val_loss", mode = "min")
        trainer = pl.Trainer(resume_from_checkpoint = TRAIN_RESUME_CHECKPOINT_PATH,
                             callbacks = checkpoint_callback, max_epochs=N_EPOCHS,
                             gpus=NUM_GPU, progress_bar_refresh_rate=30)
        trainer.fit(model,data_module)
    elif MODE == 'Inference':
        model = StorytellingModel.load_from_checkpoint(checkpoint_path = INFERENCE_BEST_CHECKPOINT_PATH, model_name = MODEL_NAME,
                                                       tokenizer = TOKENIZER, lambda1 = lambda1, lambda2 = lambda2,
                                                       lambda3 = lambda3, lambda4 = lambda4, lambda5 = lambda5)
        model.eval()
        model.freeze()
        today = date.today()
        test_dataloader = data_module.test_dataloader()
        Generate_Story = Generate_Story(model,TOKENIZER)
        Generate_Story.generate_story(test_dataloader, ORIGINDATA_SAVE_PATH + "Ours_1_" + str(today) + ".hdf5",
                                      SENEMBEDS_SAVE_PATH+ "Ours_1_" + str(today) + ".npy")




