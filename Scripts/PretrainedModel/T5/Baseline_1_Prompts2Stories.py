import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import re
import argparse
import gc
from tqdm import tqdm
from datetime import date
#import glob
#import json
#import time
#import logging
#import random
#from itertools import chain
#from string import  punctuation

import pandas as pd
#import numpy as np
import torch
#import nlp
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AdamW, T5TokenizerFast, T5ForConditionalGeneration
#from transformers import get_linear_schedule_with_warmup


def prepare_data(input_path: Path, target_path: Path):
    with open(input_path) as input_file:
        inputs = input_file.readlines()
    with open(target_path) as target_path:
        targets = target_path.readlines()

    data_rows = []

    for line_in, line_tg in zip(inputs, targets):
        input, target = line_in, line_tg
        data_rows.append({"source": input, "target": target})
    return pd.DataFrame(data_rows)


class StorytellingDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5TokenizerFast,
            source_max_token_len: int = 100,
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
        labels[labels == 0] = -100

        return dict(
            source=data_row["source"],
            target=data_row["target"],
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
            source_max_token_len: int = 100,
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


class StorytellingModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        for name, parameter in self.model.named_parameters():
            if re.match("shared.*", name) or re.match("encoder.*", name):
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)






class Generate_Story:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def generate_story(self, dataloader, save_path):
        for batch in tqdm(dataloader):
            outputs = self.model.model.generate(input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            num_beams = 3,
            max_length = 500,
            repetition_penalty = 2.5,
            length_penalty = 1.0,
            early_stopping = True,
            use_cache = True)

            with open(save_path,"a") as f:
              for generated_ids in outputs:
                  story = [self.tokenizer.decode(generated_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for generated_id in generated_ids]
                  story = " ".join(story).strip("\n")+"\n"
                  f.write(story)
            del outputs
            del story
            gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODE', type=str, choices=['Train', 'Train_Resume', 'Inference'])
    parser.add_argument('--TRAIN_SRC', type=str, default="Dataset/WritingPrompts/train.wp_source")
    parser.add_argument('--TRAIN_TGT', type=str, default="Dataset/WritingPrompts/train.wp_target")
    parser.add_argument('--VALID_SRC', type=str, default="Dataset/WritingPrompts/valid.wp_source")
    parser.add_argument('--VALID_TGT', type=str, default="Dataset/WritingPrompts/valid.wp_target")
    parser.add_argument('--TEST_SRC', type=str, default="Dataset/WritingPrompts/test.wp_source")
    parser.add_argument('--TEST_TGT', type=str, default="Dataset/WritingPrompts/test.wp_target")
    parser.add_argument('--TRAIN_RESUME_CHECKPOINT_PATH', type=str)
    parser.add_argument('--INFERENCE_BEST_CHECKPOINT_PATH', type=str)
    parser.add_argument('--INFERENCE_SAVE_PATH', type=str, default="Inference/")
    parser.add_argument('--MODEL_NAME', type=str, default="t5-small")
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--N_EPOCHS', type=int, default=20)
    parser.add_argument('--SOURCE_MAX_TOKEN_LEN', type=int, default=100)
    parser.add_argument('--TARGET_MAX_TOKEN_LEN', type=int, default=500)
    parser.add_argument('--NUM_GPU', type=int, default=1)
    parser.add_argument('--CHECKPOINTS_SAVE_PATH', type=str, default="Checkpoints/")
    args = parser.parse_args()

    MODE = args.MODE
    TRAIN_SRC = args.TRAIN_SRC
    TRAIN_TGT = args.TRAIN_TGT
    VALID_SRC = args.VALID_SRC
    VALID_TGT = args.VALID_TGT
    TEST_SRC = args.TEST_SRC
    TEST_TGT = args.TEST_TGT
    TRAIN_RESUME_CHECKPOINT_PATH = args.TRAIN_RESUME_CHECKPOINT_PATH
    INFERENCE_BEST_CHECKPOINT_PATH = args.INFERENCE_BEST_CHECKPOINT_PATH
    INFERENCE_SAVE_PATH = args.INFERENCE_SAVE_PATH
    MODEL_NAME = args.MODEL_NAME
    TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_NAME)
    BATCH_SIZE = args.BATCH_SIZE
    N_EPOCHS = args.N_EPOCHS
    SOURCE_MAX_TOKEN_LEN = args.SOURCE_MAX_TOKEN_LEN
    TARGET_MAX_TOKEN_LEN = args.TARGET_MAX_TOKEN_LEN
    NUM_GPU = args.NUM_GPU
    CHECKPOINTS_SAVE_PATH = args.CHECKPOINTS_SAVE_PATH

    if MODE == 'Train' or "Train_Resume":
        train_DataFrame = prepare_data(TRAIN_SRC, TRAIN_TGT)
        valid_DataFrame = prepare_data(VALID_SRC, VALID_TGT)
        test_DataFrame = prepare_data(VALID_SRC, VALID_TGT)
    elif MODE == 'Inference':
        train_DataFrame = prepare_data(TEST_SRC, TEST_TGT)
        valid_DataFrame = prepare_data(TEST_SRC, TEST_TGT)
        test_DataFrame = prepare_data(TEST_SRC, TEST_TGT)


    data_module = StorytellingDataModule(train_DataFrame, valid_DataFrame, test_DataFrame, TOKENIZER,
                                         batch_size = BATCH_SIZE, use_tpu = False,
                                         source_max_token_len = SOURCE_MAX_TOKEN_LEN,
                                         target_max_token_len = TARGET_MAX_TOKEN_LEN)
    data_module.setup()

    del train_DataFrame
    del valid_DataFrame
    del test_DataFrame
    gc.collect()



    if MODE == 'Train':
        model = StorytellingModel(MODEL_NAME)
        checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                              filename = 'Baseline_1_best_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                              save_top_k = 1, verbose = True, monitor = "val_loss", mode = "min")
        trainer = pl.Trainer(callbacks = checkpoint_callback, max_epochs=N_EPOCHS,
                             gpus=NUM_GPU, progress_bar_refresh_rate=30)
        trainer.fit(model,data_module)
    elif MODE == 'Train_Resume':
        model = StorytellingModel(MODEL_NAME)
        checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                              filename = 'Baseline_1_best_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                              save_top_k = 1, verbose = True, monitor = "val_loss", mode = "min")
        trainer = pl.Trainer(resume_from_checkpoint = TRAIN_RESUME_CHECKPOINT_PATH,
                             callbacks = checkpoint_callback, max_epochs=N_EPOCHS,
                             gpus=NUM_GPU, progress_bar_refresh_rate=30)
        trainer.fit(model,data_module)
    elif MODE == 'Inference':
        model = StorytellingModel.load_from_checkpoint(checkpoint_path = INFERENCE_BEST_CHECKPOINT_PATH, model_name = MODEL_NAME)
        model.eval()
        model.freeze()
        today = date.today()
        test_dataloader = data_module.test_dataloader()
        Generate_Story = Generate_Story(model,TOKENIZER)
        Generate_Story.generate_story(test_dataloader, INFERENCE_SAVE_PATH + "Baseline_1_" + str(today) + ".txt")
