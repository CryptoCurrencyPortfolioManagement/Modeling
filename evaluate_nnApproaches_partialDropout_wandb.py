import os
import math

import torch.cuda
from tqdm import tqdm
tqdm.pandas()

from datetime import datetime
import random

import pandas as pd
import pyarrow.parquet as pq

import gym
from environment import TextAssetTradingEnv
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import wandb
import GPUtil

import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from gym.envs.registration import register
register(
    id='textassettrading-v0',
    entry_point='environment:TextAssetTradingEnv',
)

# custom pytorch dataset cabable of returning items consisting of multiple elements
class CustomDataset(Dataset):
    def __init__(self, X1, X2=None, y= None):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        if self.y is not None:
            assert self.X1.shape[0] == self.y.shape[0], "Size of X1 and y does not match"
        if self.X2 is not None:
            assert self.X1.shape[0] == self.X2.shape[0], "Size of X1 and X2 does not match"
    def __len__(self):
        return self.X1.shape[0]
    def __getitem__(self, idx):
        if self.y is not None and self.X2 is not None:
            return self.X1[idx], self.X2[idx], self.y[idx]
        elif self.y is not None:
            return self.X1[idx], self.y[idx]
        else:
            return self.X1[idx]

# setup training and prediction routines
class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size, ccs, bert_model, epochs= 2, learning_rate= 1e-6, freeze_bert= False, dropout_prop= 0.5):
        # automatically choose GPU if one with enough memory is available
        if len(GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)) > 0:
            self.device = "cuda:{}".format(GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)[0])
        else:
            print('On none of the gpus enough memory is available using cpu')
            self.device = "cpu"
        #self.device = "cuda"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.tokenizer.add_tokens(ccs)
        # use pre-build bertmodel for sequence classification
        self.model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=3)
        self.freeze_bert = freeze_bert
        if self.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.ccs = ccs
        self.dropout_prop = dropout_prop

    def fit(self, X, y):
        X_text = X

        X_text = np.moveaxis(X_text, 1, 2)
        y = np.repeat(y, X_text.shape[1], axis=0)
        X_text = X_text.reshape((-1, *X_text.shape[2:]), order= "C")

        X_text_new = []
        for i, cc in enumerate(self.ccs):
            X_text_new.append(np.apply_along_axis(self._create_bert_input, 1, X_text[:, i, :], cc, self.dropout_prop))
        X_text_new = np.row_stack(X_text_new)

        mask = np.sum(X_text_new[:, 2, :], axis= 1) > 0
        y = y.reshape(-1, order= "F")[mask]
        X_text_new = X_text_new[mask]

        y[y > 0.01] = 1
        y[np.bitwise_and(y > -0.01, y < 0.01)] = 0
        y[y < -0.01] = -1

        dataloader = DataLoader(CustomDataset(X1=X_text_new, y=y), batch_size=self.batch_size, shuffle=False, drop_last= True)

        loss_history = []
        for epoch in range(self.epochs):
            self.model.train()

            for X_text_batch, y_batch in tqdm(dataloader, desc="Epoch: {}".format(epoch + 1)):
                self.optim.zero_grad()
                X_text_batch = X_text_batch.to(self.device, dtype=torch.long)
                preds_batch = self.model(input_ids=X_text_batch[:, 0, :],
                                         token_type_ids=X_text_batch[:, 1, :],
                                         attention_mask=X_text_batch[:, 2, :])
                y_batch = y_batch.to(self.device, dtype=torch.long)
                loss = self.loss_fn(preds_batch.logits, y_batch)
                loss.backward()
                loss_history.append(loss.detach().cpu().tolist())
                self.optim.step()

    def predict(self, X):
        X_text = X

        X_text = np.moveaxis(X_text, 1, 2)
        X_text = X_text.reshape((-1, *X_text.shape[2:]), order="C")

        scores = []
        for i, cc in enumerate(self.ccs):
            X_text_cc = np.apply_along_axis(self._create_bert_input, 1, X_text[:, i, :], cc)
            mask = np.sum(X_text_cc[:, 2, :], axis=1) > 0
            X_text_cc = X_text_cc[mask]
            dataloader = DataLoader(CustomDataset(X1=X_text_cc), batch_size=self.batch_size, shuffle=False)

            self.model.eval()

            temp_score = []
            with torch.no_grad():
                for X_text_batch in dataloader:
                    X_text_batch = X_text_batch.to(self.device, dtype=torch.long)
                    preds_batch = self.model(input_ids=X_text_batch[:, 0, :],
                                             token_type_ids=X_text_batch[:, 1, :],
                                             attention_mask=X_text_batch[:, 2, :])
                    temp_score.append(preds_batch.logits.detach().cpu().numpy())
            if len(temp_score) == 0:
                temp_score = 0
            else:
                temp_score = np.mean(np.row_stack(temp_score), axis= 0)
            if np.argmax(temp_score) == 1:
                temp_score = np.abs(temp_score[np.argmax(temp_score)])
            elif np.argmax(temp_score) == 2:
                temp_score = np.abs(temp_score[np.argmax(temp_score)]) * -1
            else:
                temp_score = 0
            scores.append(temp_score)
        return np.stack(scores)

    def _create_bert_input(self, data, cc, dropout=0):
        if np.sum((~np.isnan(data)).astype(int)) == 0:
            input_ids = np.concatenate(([np.nan], data))
            token_type_ids = np.array([0] * (len(data) + 1))
            attention_mask = np.concatenate(([0], (~np.isnan(data)).astype(int)))
        else:
            if random.random() > dropout:
                input_ids = np.concatenate(([data[0], self.tokenizer("token")["input_ids"][1]], data[1:]))
            else:
                input_ids = np.concatenate(([data[0], self.tokenizer(cc)["input_ids"][1]], data[1:]))
            token_type_ids = np.array([0, 1] + [0] * (len(data) - 1))
            attention_mask = np.concatenate(([1], (~np.isnan(data)).astype(int)))
        return np.array([input_ids, token_type_ids, attention_mask])

# create an agent which can be trained on a dataset and then be used in an dynamic environment to predict the next action
class NN_agent():
    def __init__(self, number_assets, texts_per_day, words_per_text, model):
        self.number_assets = number_assets
        self.model = model
        self.texts_per_day = texts_per_day
        self.words_per_text = words_per_text

    def choose_action(self, obs):
        X_temp_text = obs[:, 2*self.number_assets:]
        X_temp_text = np.split(X_temp_text, np.cumsum([int(X_temp_text.shape[1] / self.number_assets)] * self.number_assets), axis=1)[:-1]
        X_temp_text = np.array(X_temp_text)
        X_temp_text = np.moveaxis(X_temp_text, 0, 1)
        X_temp_text = np.split(X_temp_text, np.cumsum([int(X_temp_text.shape[2] / self.texts_per_day)] * self.texts_per_day), axis=2)[:-1]
        X_temp_text = np.array(X_temp_text)
        X_temp_text = np.moveaxis(X_temp_text, 0, 2)

        scores = self.model.predict(X_temp_text)

        return scores.flatten()

    def remember(self, obs, act, reward, new_state, done):
        pass

    def learn(self):
        pass

    def train(self, X, y):
        X_temp_text = X[:, self.number_assets:]
        X_temp_text = np.split(X_temp_text, np.cumsum([int(X_temp_text.shape[1]/self.number_assets)]*self.number_assets), axis= 1)[:-1]
        X_temp_text = np.array(X_temp_text)
        X_temp_text = np.moveaxis(X_temp_text, 0, 1)
        X_temp_text = np.split(X_temp_text, np.cumsum([int(X_temp_text.shape[2]/self.texts_per_day)]*self.texts_per_day), axis=2)[:-1]
        X_temp_text = np.array(X_temp_text)
        X_temp_text = np.moveaxis(X_temp_text, 0, 2)
        y_temp = y
        self.model.fit(X_temp_text, y_temp)

    def save_model(self, path):
        torch.save(self.model.model.state_dict(), path)

    def load_model(self, path):
        self.model.model.load_state_dict(torch.load(path))


# setup base configuration
ccs = ['btc', 'eth', 'xrp', 'xem', 'etc', 'ltc', 'dash', 'xmr', 'strat', 'xlm']

train_start_date = pd.to_datetime("2017-06-01").date()
train_end_date = pd.to_datetime("2020-05-31").date()

results_df = pd.DataFrame()

price_df = pq.ParquetDataset("./data/Price.parquet", validate_schema=False, filters=[('cc', 'in', ccs)])
price_df = price_df.read(columns=["price"]).to_pandas()
price_df["date"] = pd.to_datetime(price_df["date"]).apply(lambda x: x.date())
price_df = price_df.pivot(index='date', columns='cc', values='price')

warm_up_window = 180
rolling_windows_size = 360
rolling_windows_shift = 90


if __name__ == "__main__":
    with wandb.init() as run:
        param = wandb.config
        #param = {"freeze_bert": True, "learning_rate": 0.0008, "model": "bert-base-uncased", "source": "CoinTelegraph", "window_size": 1, "dropout": 0, "epochs": 3}

        # load Data
        text_df = pq.ParquetDataset("data/Content.parquet", validate_schema=False, filters=[('cc', 'in', ccs), ('source', '=', param["source"])])
        if param["model"] == "yiyanghkust/finbert-tone":
            text_df = text_df.read(columns=["content_finbertToken"]).to_pandas()
        else:
            text_df = text_df.read(columns=["content_bertToken"]).to_pandas()
        text_df["date"] = pd.to_datetime(text_df["date"]).apply(lambda x: x.date())
        text_df = text_df.set_index("date").drop("source", axis=1)

        date_range = [x.date() for x in pd.date_range(train_start_date, train_end_date)]

        assert len(date_range) > warm_up_window + rolling_windows_size, "Given data has too few observations for defined rolling windows"

        # create windows
        windows = []
        window_start_idx = warm_up_window
        window_end_idx = warm_up_window+rolling_windows_size
        while window_end_idx <= len(date_range):
            windows.append((date_range[window_start_idx], date_range[window_end_idx]))
            window_start_idx += rolling_windows_shift
            window_end_idx = window_start_idx + rolling_windows_size

        metrics = {"cumReturn": [],
                  "sharpeRatio": [],
                  "sortinoRatio": [],
                  "calmarRatio": [],
                  "mdd": []}

        for i, (test_roll_start_date, test_roll_end_date) in enumerate(windows):

            # create environments and setup models
            test_env = gym.make('textassettrading-v0',
                                price_df=price_df,
                                window_size=param["window_size"],
                                text_df=text_df,
                                timeframe_bound=(test_roll_start_date, test_roll_end_date),
                                reward= None,
                                max_texts_per_day= 30,
                                max_words_per_text= 20,
                                turbulence_threshold=np.inf,
                                trade_fee_bid_percent= 0.005,
                                trade_fee_ask_percent= 0.005,
                                force_trades= True,
                                provide_actions= False)

            my_model = ModelWrapper(batch_size= 128, ccs= test_env.labels, bert_model= param["model"], learning_rate= param["learning_rate"], freeze_bert= param["freeze_bert"], dropout_prop= param["dropout"], epochs= param["epochs"])
            agent = NN_agent(test_env.stock_dim, texts_per_day= test_env.texts_per_day, words_per_text=test_env.words_per_text, model= my_model)

            # only train model if no trained model can be loaded
            model_path = "./models/model_{}".format("_".join([str(x).replace("/","-") for x in list(dict(param).values())])) + "_CV{}.pt".format(i)
            if os.path.exists(model_path):
                agent.load_model(model_path)
            else:
                train_env = gym.make('textassettrading-v0',
                                    price_df=price_df,
                                    window_size=1,
                                    text_df=text_df,
                                    timeframe_bound=(train_start_date, test_roll_start_date),
                                    reward=test_env.reward_type,
                                    max_texts_per_day=test_env.texts_per_day,
                                    max_words_per_text=test_env.words_per_text,
                                    turbulence_threshold=np.inf,
                                    trade_fee_bid_percent=test_env.trade_fee_ask_percent,
                                    trade_fee_ask_percent=test_env.trade_fee_bid_percent,
                                    force_trades=test_env.force_trades,
                                    provide_actions=test_env.provide_actions)

                # train model on static training data
                train_X, train_y = train_env._get_dataframes()
                agent.train(train_X, train_y)
                agent.save_model(model_path)

            obs = test_env.reset()
            done = False
            score = 0
            m = 0

            # evaluate the model on the dynamic environment
            while not done:
                act = agent.choose_action(obs)
                new_state, reward, done, info = test_env.step(act)
                agent.remember(obs, act, reward, new_state, done)
                agent.learn()
                score += reward
                obs = new_state
                #if m % 50 == 0 or done:
                #    test_env.render(path= "./plots/dictApproaches/{}_{}_{}_{}_{}.png".format(num_cols[i], window_size[j], assume_reversion[k], "-".join(sources[l]), j))
                m += 1

            #test_env.render_all(path= "./plots/{}_{}_{}_{}_CV".format(*[str(x).replace("/","-") for x in list(dict(param).values())]) + "{}.png".format(i))

            # save generated data from model evaluation
            pd.DataFrame(test_env.history).to_excel("./results/{}_CV".format(run.name) + "{}.xlsx".format(i), index=False)

            wandb_dict = test_env._total_ratios
            wandb_dict["cumReturn"] = test_env._total_return
            wandb_dict = {k+'_CV': v for k, v in wandb_dict.items()}
            wandb.log(wandb_dict)

            for key in metrics.keys():
                if key == "cumReturn":
                    metrics[key].append(test_env._total_return)
                else:
                    metrics[key].append(test_env._total_ratios[key])

        save_dict = {}
        for key in metrics.keys():
            save_dict[key] = np.nanmean(metrics[key])
            save_dict[key + "_std"] = np.nanstd(metrics[key])
            save_dict[key + "_q1"] = np.nanquantile(metrics[key], 0.25)
            save_dict[key + "_q3"] = np.nanquantile(metrics[key], 0.75)
            save_dict[key + "_min"] = np.nanmin(metrics[key])
            save_dict[key + "_max"] = np.nanmax(metrics[key])
            save_dict[key + "_med"] = np.nanmedian(metrics[key])

        wandb.log(save_dict)