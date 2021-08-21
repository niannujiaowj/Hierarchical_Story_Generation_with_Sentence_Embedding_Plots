import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import re
import argparse
from tqdm import tqdm
from datetime import date
import gc


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# NEED modified transformers from https://github.com/niannujiaowj/transformers.git
from transformers import AdamW, T5TokenizerFast, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from utils import CrossEntropyLoss


def prepare_data(input_plot_path: Path, input_senembedding_path: Path,  target_path: Path):
    with open(input_plot_path) as input_file:
        inputs_plot = input_file.readlines()

    print("Start loading data...")
    inputs_senembedding = np.load(input_senembedding_path, allow_pickle=True)

    with open(target_path) as target_path:
        targets = target_path.readlines()

    print("Start generating data...")
    data_rows = []
    start_story = 0
    if re.match(".*npz$", input_senembedding_path):
        for file in tqdm(inputs_senembedding.files):
            end_story = len(inputs_senembedding[file]) + start_story
            for line_in_plot, line_in_senembeddings, line_tg in zip(inputs_plot[start_story:end_story], inputs_senembedding[file], targets[start_story:end_story]):
                data_rows.append({"plots": line_in_plot, "senembeddings": line_in_senembeddings,
                                  "senembedding_tokens": "<SEN> "* len(line_in_senembeddings), "targets": line_tg})
            start_story += len(file)
    else:
        for line_in_plot, line_in_senembeddings, line_tg in zip(inputs_plot, inputs_senembedding, targets):
          data_rows.append({"plots": line_in_plot,  "senembeddings": line_in_senembeddings,
                            "senembedding_tokens": "<SEN> "* len(line_in_senembeddings), "targets": line_tg})

    del targets
    del inputs_plot
    del inputs_senembedding
    gc.collect()

    return pd.DataFrame(data_rows)


class StorytellingDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5TokenizerFast,
            source_plot_max_token_len: int = 1000,
            source_senembedding_max_token_len: int = 50,
            target_max_token_len: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_plot_max_token_len = source_plot_max_token_len
        self.source_senembedding_max_token_len = source_senembedding_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        plots_encoding = self.tokenizer(
            data_row["plots"],
            max_length=self.source_plot_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False,  # don't add <eos> token
            return_tensors="pt"
        )

        senembedding_tokens_encoding = self.tokenizer(
            data_row["senembedding_tokens"],
            max_length=self.source_senembedding_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,  # add <eos> token
            return_tensors="pt"
        )

        targets_encoding = self.tokenizer(
            data_row["targets"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = targets_encoding["input_ids"]
        labels[labels == 0] = -100

        plot_ids = plots_encoding["input_ids"].flatten()
        senembedding_tokens_ids = senembedding_tokens_encoding["input_ids"].flatten()

        try:
            plot_ids_zero_indx = (plot_ids == 0).nonzero(as_tuple=True)[0][0]
        except:
            plot_ids_zero_indx = -1

        try:
            senembedding_tokens_ids_zero_indx = (senembedding_tokens_ids == 0).nonzero(as_tuple=True)[0][0]
        except:
            senembedding_tokens_ids_zero_indx = -1

        plot_ids = plot_ids[:plot_ids_zero_indx] if plot_ids_zero_indx != -1 else plot_ids
        senembedding_tokens_ids = senembedding_tokens_ids[
                                  :senembedding_tokens_ids_zero_indx] if senembedding_tokens_ids_zero_indx != -1 else senembedding_tokens_ids
        number_of_zero_pad = self.source_plot_max_token_len \
                             + self.source_senembedding_max_token_len \
                             - len(plot_ids) \
                             - len(senembedding_tokens_ids)
        input_ids = torch.cat([plot_ids, senembedding_tokens_ids, torch.zeros(number_of_zero_pad)]).type(
            dtype=torch.long)
        attention_mask = torch.where(input_ids == 0, input_ids, 1)

        sen_id = self.tokenizer.encode("<SEN>")[0]
        senembeddings_orig = torch.tensor(data_row["senembeddings"]).view(-1, 768)
        # senembeddings_orig = torch.from_numpy(data_row["senembeddings"])
        senembeddings = torch.zeros(len(input_ids), 768)
        location_sen = (input_ids == sen_id).nonzero(as_tuple=False)
        location_sen = [index.item() for _ in location_sen for index in _]
        for index, n in zip(location_sen, range(0, len(senembeddings_orig))):
            senembeddings[index, :] = senembeddings_orig[n, :]

        return dict(
            source_plot=data_row["plots"],
            source_senembedding_token=data_row["senembedding_tokens"],
            target=data_row["targets"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            senembeddings=senembeddings,
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
            source_plot_max_token_len: int = 1000,
            source_senembedding_max_token_len: int = 50,
            target_max_token_len: int = 500,
            use_tpu=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_DataFrame = train_DataFrame
        self.valid_DataFrame = valid_DataFrame
        self.test_DataFrame = test_DataFrame
        self.tokenizer = tokenizer
        self.source_plot_max_token_len = source_plot_max_token_len
        self.source_senembedding_max_token_len = source_senembedding_max_token_len
        self.target_max_token_len = target_max_token_len
        self.use_tpu = use_tpu
        self.sampler_train = None
        self.sampler_valid = None
        self.sampler_test = None

    def setup(self):
        self.train_dataset = StorytellingDataset(self.train_DataFrame, self.tokenizer, self.source_plot_max_token_len,
                                                 self.source_senembedding_max_token_len, self.target_max_token_len)
        self.valid_dataset = StorytellingDataset(self.valid_DataFrame, self.tokenizer, self.source_plot_max_token_len,
                                                 self.source_senembedding_max_token_len, self.target_max_token_len)
        self.test_dataset = StorytellingDataset(self.test_DataFrame, self.tokenizer, self.source_plot_max_token_len,
                                                self.source_senembedding_max_token_len, self.target_max_token_len)
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


class MyModel(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(MyModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True,
                                                                output_hidden_states=True).to(self.device)
        self.model.resize_token_embeddings(len(tokenizer))
        self.sen_id = tokenizer.encode("<SEN>")[0]
        self.tokenizer = tokenizer

        self.senembedding_project_layer = nn.Linear(768, 512, bias=False).to(self.device)

    def forward(self, input_ids, senembeddings, attention_mask, labels=None):
        '''
        input_ids (batch size, max_src_length)
        attention_mask (batch size, max_src_length)
        labels (batch size, max_tgt_length)
        senembeddings (batch size, max_src_length, 768)
        '''

        senembeddings = self.senembedding_project_layer(senembeddings)

        plot_ids = torch.where(input_ids != self.sen_id, input_ids, 0)
        plot_onehot = F.one_hot(plot_ids, num_classes=len(self.tokenizer)).view(-1, len(self.tokenizer)).type(
            dtype=torch.float)
        for name, para in self.model.named_parameters():
            if name == "shared.weight":
                T5embeddings = para  # (vocab size, 512)
                break
        wordembeddings = torch.matmul(plot_onehot, T5embeddings).view(input_ids.size()[0], -1, 512)

        jointwordembeddings = torch.where(senembeddings != 0, senembeddings, wordembeddings)
        # print("jointwordembeddings",jointwordembeddings,jointwordembeddings.size())
        # print("attention_mask",attention_mask, attention_mask.size())
        # print("labels",labels, labels.size())

        outputs = self.model(inputs_embeds=jointwordembeddings, attention_mask=attention_mask, labels=labels)

        loss_fn = CrossEntropyLoss(self.tokenizer)
        loss = loss_fn(outputs.logits, labels)
        return loss

    def generate(self, input_ids, senembeddings, attention_mask,
                 num_beams=3, max_length=500, repetition_penalty=2.5,
                 length_penalty=1.0, early_stopping=True, use_cache=True):
        senembeddings = self.senembedding_project_layer(senembeddings.to(self.device)).to(self.device)

        plot_ids = torch.where(input_ids.to(self.device) != self.sen_id, input_ids.to(self.device), 0).to(self.device)
        plot_onehot = F.one_hot(plot_ids, num_classes=len(self.tokenizer)).view(-1, len(self.tokenizer)).type(
            dtype=torch.float).to(self.device)
        for name, para in self.model.named_parameters():
            if name == "shared.weight":
                T5embeddings = para.to(self.device)  # (vocab size, 512)
                break
        wordembeddings = torch.matmul(plot_onehot, T5embeddings).view(input_ids.size()[0], -1, 512).to(self.device)

        jointwordembeddings = torch.where(senembeddings != 0, senembeddings, wordembeddings).to(self.device)

        outputs = self.model.generate(inputs_embeds=jointwordembeddings, attention_mask=attention_mask.to(self.device),
                                      # modified generate.py
                                      num_beams=num_beams, max_length=max_length, repetition_penalty=repetition_penalty,
                                      length_penalty=length_penalty, early_stopping=early_stopping,
                                      use_cache=use_cache).to(self.device)

        return outputs


class StorytellingModel(pl.LightningModule):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.model = MyModel(model_name, tokenizer)
        for name, parameter in self.model.named_parameters():
            if re.match("shared.*", name) or re.match("encoder.*", name):
                parameter.requires_grad = False

    def forward(self, input_ids, senembeddings, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, senembeddings=senembeddings, attention_mask=attention_mask,
                             labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        senembeddings = batch["senembeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, senembeddings, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        senembeddings = batch["senembeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # print("input_ids",input_ids,input_ids.size())
        # print("senembeddings",senembeddings,senembeddings.size())
        # print("attention_mask",attention_mask,attention_mask.size())
        # print("labels",labels,labels.size())
        loss = self(input_ids, senembeddings, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        senembeddings = batch["senembeddings"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, senembeddings, attention_mask, labels)
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
            senembeddings = batch["senembeddings"],
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
    parser.add_argument('--TRAIN_SRC_PLOTS', type=str, default="Dataset/Plots/train")
    parser.add_argument('--TRAIN_SRC_SEN', type=str, default="Dataset/SenEmbedding/train.npz")
    parser.add_argument('--TRAIN_TGT', type=str, default="Dataset/WritingPrompts/train.wp_source")
    parser.add_argument('--VALID_SRC_PLOTS', type=str, default="Dataset/Plots/valid")
    parser.add_argument('--VALID_SRC_SEN', type=str, default="Dataset/SenEmbedding/valid.npy")
    parser.add_argument('--VALID_TGT', type=str, default="Dataset/WritingPrompts/valid.wp_source")
    parser.add_argument('--TEST_SRC_PLOTS', type=str, default="Dataset/Plots/test")
    parser.add_argument('--TEST_SRC_SEN', type=str, default="Dataset/SenEmbedding/test.npz")
    parser.add_argument('--TEST_TGT', type=str, default="Dataset/WritingPrompts/test.wp_source")
    parser.add_argument('--TRAIN_RESUME_CHECKPOINT_PATH', type=str)
    parser.add_argument('--INFERENCE_BEST_CHECKPOINT_PATH', type=str)
    parser.add_argument('--INFERENCE_SAVE_PATH', type=str, default="Inference/")
    parser.add_argument('--MODEL_NAME', type=str, default="t5-small")
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--N_EPOCHS', type=int, default=20)
    parser.add_argument('--SOURCE_PLOT_MAX_TOKEN_LEN', type=int, default=500)
    parser.add_argument('--SOURCE_SENEMBEDDING_MAX_TOKEN_LEN', type=int, default=50)
    parser.add_argument('--TARGET_MAX_TOKEN_LEN', type=int, default=500)
    parser.add_argument('--NUM_GPU', type=int, default=1)
    parser.add_argument('--CHECKPOINTS_SAVE_PATH', type=str, default="Checkpoints/")
    args = parser.parse_args()

    MODE = args.MODE
    MODEL_NAME = args.MODEL_NAME
    TOKENIZER = T5TokenizerFast.from_pretrained(MODEL_NAME)
    TOKENIZER.add_tokens("<SEN>")
    TRAIN_SRC_PLOTS = args.TRAIN_SRC_PLOTS
    TRAIN_SRC_SEN = args.TRAIN_SRC_SEN
    TRAIN_TGT = args.TRAIN_TGT
    VALID_SRC_PLOTS = args.VALID_SRC_PLOTS
    VALID_SRC_SEN = args.VALID_SRC_SEN
    VALID_TGT = args.VALID_TGT
    TEST_SRC_PLOTS = args.TEST_SRC_PLOTS
    TEST_SRC_SEN = args.TEST_SRC_SEN
    TEST_TGT = args.TEST_TGT
    TRAIN_RESUME_CHECKPOINT_PATH = args.TRAIN_RESUME_CHECKPOINT_PATH
    INFERENCE_BEST_CHECKPOINT_PATH = args.INFERENCE_BEST_CHECKPOINT_PATH
    INFERENCE_SAVE_PATH = args.INFERENCE_SAVE_PATH
    BATCH_SIZE = args.BATCH_SIZE
    N_EPOCHS = args.N_EPOCHS
    SOURCE_PLOT_MAX_TOKEN_LEN = args.SOURCE_PLOT_MAX_TOKEN_LEN
    SOURCE_SENEMBEDDING_MAX_TOKEN_LEN = args.SOURCE_SENEMBEDDING_MAX_TOKEN_LEN
    TARGET_MAX_TOKEN_LEN = args.TARGET_MAX_TOKEN_LEN
    NUM_GPU = args.NUM_GPU
    CHECKPOINTS_SAVE_PATH = args.CHECKPOINTS_SAVE_PATH



    if MODE == 'Train' or "Train_Resume":
        train_DataFrame = prepare_data(TRAIN_SRC_PLOTS, TRAIN_SRC_SEN, TRAIN_TGT)
        valid_DataFrame = prepare_data(VALID_SRC_PLOTS, VALID_SRC_SEN, VALID_TGT)
        test_DataFrame = prepare_data(VALID_SRC_PLOTS, VALID_SRC_SEN, VALID_TGT)
    elif MODE == 'Inference':
        train_DataFrame = prepare_data(TEST_SRC_PLOTS, TEST_SRC_SEN, TEST_TGT)
        valid_DataFrame = prepare_data(TEST_SRC_PLOTS, TEST_SRC_SEN, TEST_TGT)
        test_DataFrame = prepare_data(TEST_SRC_PLOTS, TEST_SRC_SEN, TEST_TGT)




    data_module = StorytellingDataModule(train_DataFrame, valid_DataFrame, test_DataFrame, tokenizer = TOKENIZER,
                                         batch_size=BATCH_SIZE, use_tpu=False,
                                         source_plot_max_token_len = SOURCE_PLOT_MAX_TOKEN_LEN,
                                         source_senembedding_max_token_len = SOURCE_SENEMBEDDING_MAX_TOKEN_LEN,
                                         target_max_token_len=TARGET_MAX_TOKEN_LEN)
    data_module.setup()


    if MODE == 'Train':
        model = StorytellingModel(MODEL_NAME, TOKENIZER)
        checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                              filename = 'Ours_2_best_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                              save_top_k = 1, verbose = True, monitor = "val_loss", mode = "min")
        trainer = pl.Trainer(callbacks = checkpoint_callback, max_epochs=N_EPOCHS,
                             gpus=NUM_GPU, progress_bar_refresh_rate=30)
        trainer.fit(model,data_module)
    elif MODE == 'Train_Resume':
        model = StorytellingModel(MODEL_NAME, TOKENIZER)
        checkpoint_callback = ModelCheckpoint(dirpath = CHECKPOINTS_SAVE_PATH,
                                              filename = 'Ours_2_best_checkpoint_{epoch:02d}_{val_loss:.2f}',
                                              save_top_k = 1, verbose = True, monitor = "val_loss", mode = "min")
        trainer = pl.Trainer(resume_from_checkpoint = TRAIN_RESUME_CHECKPOINT_PATH,
                             callbacks = checkpoint_callback, max_epochs=N_EPOCHS,
                             gpus=NUM_GPU, progress_bar_refresh_rate=30)
        trainer.fit(model,data_module)
    elif MODE == 'Inference':
        model = StorytellingModel.load_from_checkpoint(checkpoint_path = INFERENCE_BEST_CHECKPOINT_PATH,
                                                       model_name = MODEL_NAME, tokenizer = TOKENIZER)
        model.eval()
        model.freeze()
        today = date.today()
        test_dataloader = data_module.test_dataloader()
        print("Note that inference mode requires modified transformers from https://github.com/niannujiaowj/transformers.git")
        Generate_Story = Generate_Story(model,TOKENIZER)
        Generate_Story.generate_story(test_dataloader, INFERENCE_SAVE_PATH + "Ours_2_" + str(today) + ".txt")



