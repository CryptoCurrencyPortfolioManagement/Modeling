import os
import math

import torch.cuda
from tqdm import tqdm
tqdm.pandas()

from datetime import datetime

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

import GPUtil
import wandb

import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from gensim.models import KeyedVectors
fasttext_model = KeyedVectors.load("./other/crawl-300d-2M.model")

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

# layer which allows to randomly drop certain elements from defined indeces of the input tensor
class Dropout_partial_binary(nn.Module):
    def __init__(self, include_index: list, p: float = 0.5):
        super().__init__()
        self.include_index = include_index
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, X):
        X_to_modify = X[:, :, self.include_index]

        if self.training:
            X_power = torch.sum(X_to_modify, dim= -1)
            X_to_modify[torch.rand(X_to_modify.shape)<self.p] = 0
            X_to_modify = torch.add(X_to_modify, self._reshape_fortran(((X_power - torch.sum(X_to_modify, dim= -1))/X_to_modify.shape[-1]).repeat(1, X_to_modify.shape[-1]), X_to_modify.shape))

        X[:, :, self.include_index] = X_to_modify

        return X

    def _reshape_fortran(self, x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

# implements the attention layer as proposed by Vaswani et al. (2017)
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, batch_first):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.batch_first = batch_first

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        if self.batch_first:
            keys = torch.moveaxis(keys, 0, 1)
            values = torch.moveaxis(values, 0, 1)

        query = query.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0, 1).transpose(1, 2)  # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = torch.nn.functional.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

# model to combine aggregated w2v embedding over time
class sumW2VRecurrentPriceClassifier(nn.Module):
    def __init__(self, use_price, number_assets= 10, embed_dim=301, portfolio_dim= 32, use_attention= False, rnn_layers = 1, linear_layers= 1, portfolio_layers= 1, dropout_prop= 0.5):
        super(sumW2VRecurrentPriceClassifier, self).__init__()
        self.use_price = use_price
        self.use_attention = use_attention
        self.rnn_layers = rnn_layers
        self.linear_layers = linear_layers
        self.portfolio_layers = portfolio_layers
        self.number_assets = number_assets
        self.dropout_prop = dropout_prop

        self.partial_dropout = Dropout_partial_binary(list(range(self.number_assets)), self.dropout_prop)

        if self.use_price:
            self.gru = nn.GRU(embed_dim + 1, embed_dim, self.rnn_layers, bidirectional= False, batch_first= True)
        else:
            self.gru = nn.GRU(embed_dim, embed_dim, self.rnn_layers, bidirectional= False, batch_first= True)

        if self.use_attention:
            self.attention = Attention(embed_dim, embed_dim, embed_dim, batch_first= True)

        self.out = nn.ModuleList()
        for i in range(self.linear_layers - 1):
            self.out.append(nn.Linear(embed_dim, embed_dim))
        self.out.append(nn.Linear(embed_dim, 1))

        # implement portfolio layers to combine scores of different assets
        self.portfolio = nn.ModuleList()
        self.portfolio.append(nn.Linear(self.number_assets, portfolio_dim))
        for i in range(self.portfolio_layers - 1):
            self.portfolio.append(nn.Linear(len(self.portfolio_dim), portfolio_dim))
        self.portfolio.append(nn.Linear(portfolio_dim, self.number_assets))

    def forward(self, text, price= None):

        orig_shape = text.shape[:2]
        text = torch.reshape(text, (-1, *text.shape[2:]))
        if self.use_price:
            price = torch.reshape(price, (-1, *price.shape[2:]))

        text = self.partial_dropout(text)

        if self.use_price:
            combined = torch.cat((text, torch.unsqueeze(price, -1)), -1)
        else:
            combined = text

        outputs, hidden = self.gru(combined)
        if isinstance(hidden, tuple):
            hidden = hidden[1]
        hidden = hidden[-1]

        if self.use_attention:
            energy, hidden = self.attention(hidden, outputs, outputs)

        for layer in self.out:
            hidden = layer(hidden)

        hidden = torch.reshape(hidden, (orig_shape[0], -1))

        for layer in self.portfolio:
            hidden = layer(hidden)

        return torch.squeeze(hidden)

# setup training and prediction routines
class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size, ccs, use_price= True, epochs= 2, learning_rate= 1e-5, use_attention= False, rnn_layers= 1, linear_layers= 1, portfolio_layers= 1, portfolio_dim=32, dropout_prop= 0.5, device= "cpu"):
        self.device = device
        self.model = sumW2VRecurrentPriceClassifier(use_price, embed_dim= 300 + len(ccs), use_attention= use_attention, rnn_layers= rnn_layers, linear_layers= linear_layers, portfolio_layers= portfolio_layers, portfolio_dim= portfolio_dim, dropout_prop= dropout_prop)
        self.model.to(self.device)
        self.loss_fn = nn.MSELoss().to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.use_price = use_price
        self.epochs = epochs
        self.ccs = ccs

    def fit(self, X, y):
        if self.use_price:
            X_text, X_price = X
        else:
            X_text = X
            X_price = None

        X_text_new = []
        for i, cc in enumerate(self.ccs):
            X_text_new_temp = []
            for j in range(X_text.shape[2]):
                X_text_new_temp.append(np.apply_along_axis(self._create_w2v_input, 1, X_text[:, i, j, :], cc))
            X_text_new.append(np.stack(X_text_new_temp, axis=1))
        X_text_new = np.stack(X_text_new, axis=1)


        y_new = y-np.repeat(np.expand_dims(y.mean(axis= 1), axis=1), len(self.ccs),axis= 1)
        y_new = np.divide(y_new, np.repeat(np.expand_dims(y_new.max(axis= 1), axis=1), len(self.ccs), axis= 1))

        X_text_new[np.isnan(X_text_new)] = 0

        if self.use_price:
            dataloader = DataLoader(CustomDataset(X1=X_text_new, X2= X_price, y=y_new), batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(CustomDataset(X1=X_text_new, y=y_new), batch_size=self.batch_size, shuffle=True, drop_last=True)

        loss_history = []
        for epoch in range(self.epochs):
            self.model.train()

            if self.use_price:
                for X_text_batch, X_price_batch, y_batch in tqdm(dataloader, desc="Epoch: {}".format(epoch + 1)):
                    self.optim.zero_grad()
                    X_text_batch = X_text_batch.to(self.device, dtype=torch.float)
                    X_price_batch = X_price_batch.to(self.device, dtype=torch.float)
                    preds_batch = self.model(X_text_batch, X_price_batch)
                    y_batch = y_batch.to(self.device, dtype=torch.float)
                    loss = self.loss_fn(preds_batch, y_batch)
                    loss.backward()
                    loss_history.append(loss.detach().cpu().tolist())
                    self.optim.step()
            else:
                for X_text_batch, y_batch in tqdm(dataloader, desc="Epoch: {}".format(epoch + 1)):
                    self.optim.zero_grad()
                    X_text_batch = X_text_batch.to(self.device, dtype=torch.float)
                    preds_batch = self.model(X_text_batch)
                    y_batch = y_batch.to(self.device, dtype=torch.float)
                    loss = self.loss_fn(preds_batch, y_batch)
                    loss.backward()
                    loss_history.append(loss.detach().cpu().tolist())
                    self.optim.step()

    def predict(self, X):
        if self.use_price:
            X_text, X_price = X
        else:
            X_text = X
            X_price = None

        X_text_new = []
        for i, cc in enumerate(self.ccs):
            X_text_new_temp = []
            for j in range(X_text.shape[2]):
                X_text_new_temp.append(np.apply_along_axis(self._create_w2v_input, 1, X_text[:, i, j, :], cc))
            X_text_new.append(np.stack(X_text_new_temp, axis=1))
        X_text_new = np.stack(X_text_new, axis=1)
        X_text_new[np.isnan(X_text_new)] = 0
        X_text_new = torch.tensor(X_text_new).to(self.device, dtype=torch.float)

        if self.use_price:
            X_price = torch.tensor(X_price).to(self.device, dtype=torch.float)

        self.model.eval()
        with torch.no_grad():
            if self.use_price:
                scores = torch.tanh(self.model(X_text_new, X_price)).detach().cpu().numpy()
            else:
                scores = torch.tanh(self.model(X_text_new)).detach().cpu().numpy()

        return scores

    def _create_w2v_input(self, data, cc):
        cc_array = np.zeros(len(self.ccs))
        if np.sum((~np.isnan(data)).astype(int)) == 0:
            cc_array[:] = np.nan
        else:
            cc_ind = self.ccs.index(cc)
            cc_array[cc_ind] = 1
        return np.concatenate((cc_array, data))

# create an agent which can be trained on a dataset and then be used in an dynamic environment to predict the next action
class NN_agent:
    def __init__(self, number_assets, embed_size, model, use_price):
        self.use_price = use_price
        self.number_assets = number_assets
        self.model = model
        self.embed_size = embed_size
        self.size_lag = self.embed_size * self.number_assets + self.number_assets

    def choose_action(self, obs):
        X = obs[:, self.number_assets:]
        X = np.reshape(X, (1, -1), order= "C")
        lag = int(X.shape[1] / self.size_lag)
        text_ind = np.array([])
        for i in range(lag):
            text_ind = np.append(text_ind, np.r_[(self.size_lag * i) + self.number_assets: (self.size_lag * i) + self.number_assets + (self.number_assets * self.embed_size)])
        text_ind = text_ind.astype(int)
        X_temp_text = X[:, text_ind]
        temp = []
        for i in range(self.number_assets):
            temp_ind = np.array([])
            for j in range(lag):
                #temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.number_assets) * j + (self.embed_size * i): (self.size_lag - self.number_assets) * j + (self.embed_size * i) + self.embed_size])
                temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.number_assets) * j + i: (self.size_lag - self.number_assets) * (j + 1): self.number_assets])
            temp_ind = temp_ind.astype(int)
            temp.append(X_temp_text[:, temp_ind])
        X_temp_text = np.array(temp)
        del temp
        X_temp_text = np.moveaxis(X_temp_text, 0, 1)
        #X_temp_text = np.split(X_temp_text, np.cumsum([int(X_temp_text.shape[2] / self.embed_size)] * self.embed_size), axis=2)[:-1]
        X_temp_text = np.split(X_temp_text, np.cumsum([self.embed_size] * int(X_temp_text.shape[-1] / self.embed_size)), axis=2)[:-1]
        X_temp_text = np.array(X_temp_text)
        X_temp_text = np.moveaxis(X_temp_text, 0, 2)

        if self.use_price:
            price_ind = np.array([])
            for i in range(lag):
                price_ind = np.append(price_ind, np.r_[(self.size_lag * i): (self.size_lag * i) + self.number_assets])
            price_ind = price_ind.astype(int)
            X_temp_price = X[:, price_ind]
            X_temp_price = np.reshape(X_temp_price, [*X_temp_price.shape[:-1], int(self.number_assets), int(lag)])
            scores = self.model.predict((X_temp_text, X_temp_price))
        else:
            scores = self.model.predict(X_temp_text)

        return scores.flatten()

    def remember(self, obs, act, reward, new_state, done):
        pass

    def learn(self):
        pass

    def train(self, X, y):
        lag = int(X.shape[1]/self.size_lag)
        text_ind = np.array([])
        for i in range(lag):
            text_ind = np.append(text_ind, np.r_[(self.size_lag * i) + self.number_assets : (self.size_lag * i) + self.number_assets + (self.number_assets * self.embed_size)])
        text_ind = text_ind.astype(int)
        X_temp_text = X[:, text_ind]
        temp = []
        for i in range(self.number_assets):
            temp_ind = np.array([])
            for j in range(lag):
                #temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.number_assets) * j + (self.embed_size * i) : (self.size_lag - self.number_assets) * j + (self.embed_size * i) + self.embed_size])
                temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.number_assets) * j + i: (self.size_lag - self.number_assets) * (j + 1): self.number_assets])
            temp_ind = temp_ind.astype(int)
            temp.append(X_temp_text[:, temp_ind])
        X_temp_text = np.array(temp)
        del temp
        X_temp_text = np.moveaxis(X_temp_text, 0, 1)
        #X_temp_text = np.split(X_temp_text, np.cumsum([int(X_temp_text.shape[-1]/self.embed_size)]*self.embed_size), axis=2)[:-1]
        X_temp_text = np.split(X_temp_text, np.cumsum([self.embed_size] * int(X_temp_text.shape[-1] / self.embed_size)), axis=2)[:-1]
        X_temp_text = np.array(X_temp_text)
        X_temp_text = np.moveaxis(X_temp_text, 0,2)
        y_temp = y

        if self.use_price:
            price_ind = np.array([])
            for i in range(lag):
                price_ind = np.append(price_ind, np.r_[(self.size_lag * i): (self.size_lag * i) + self.number_assets])
            price_ind = price_ind.astype(int)
            X_temp_price = X[:, price_ind]
            X_temp_price = np.reshape(X_temp_price, [*X_temp_price.shape[:-1], int(self.number_assets), int(lag)])
            self.model.fit((X_temp_text, X_temp_price), y_temp)
        else:
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
        # automatically choose GPU if one with enough memory is available
        if len(GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)) > 0:
            device = "cuda:{}".format(GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)[0])
        else:
            print('On none of the gpus enough memory is available using cpu')
            device = "cpu"

        param = wandb.config
        #param = {"source": "CoinTelegraph", "window_size": 3, "learning_rate": 0.0002651, "use_attention": False, "rnn_layers": 2, "linear_layers": 3, "use_price": False, "dropout": 0.5, "epochs": 2, "portfolio_layers": 1, "portfolio_dim": 32}

        # load Data
        text_df = pq.ParquetDataset("data/Content.parquet", validate_schema=False, filters=[('cc', 'in', ccs), ('source', '=', param["source"])])
        text_df = text_df.read(columns=["content_w2vSum"]).to_pandas()
        text_df["date"] = pd.to_datetime(text_df["date"]).apply(lambda x: x.date())
        text_df = text_df.set_index("date").drop("source", axis=1)
        text_df = pd.concat((text_df["content_w2vSum"].progress_apply(pd.Series), text_df.iloc[:, -1:]), axis= 1)
        text_df.columns = [str(x) for x in text_df.columns]
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
                                num_df=text_df,
                                timeframe_bound=(test_roll_start_date, test_roll_end_date),
                                reward=None,
                                max_texts_per_day=30,
                                max_words_per_text=20,
                                turbulence_threshold=np.inf,
                                trade_fee_bid_percent=0.005,
                                trade_fee_ask_percent=0.005,
                                force_trades=True,
                                provide_actions=False)

            my_model = ModelWrapper(batch_size= 64, ccs= test_env.labels, learning_rate=param["learning_rate"], use_attention= param["use_attention"], rnn_layers= param["rnn_layers"], linear_layers= param["linear_layers"], portfolio_layers= param["portfolio_layers"], portfolio_dim= param["portfolio_dim"], dropout_prop= param["dropout"], epochs= param["epochs"], device= device, use_price= param["use_price"])
            agent = NN_agent(test_env.stock_dim, embed_size= 300, model= my_model, use_price= param["use_price"])

            # only train model if no trained model can be loaded
            model_path = "./models/model_{}".format("_".join([str(x).replace("/","-") for x in list(dict(param).values())])) + "_CV{}.pt".format(i)
            if os.path.exists(model_path):
                agent.load_model(model_path)
            else:
                train_env = gym.make('textassettrading-v0',
                                     price_df=price_df,
                                     window_size=test_env.window_size,
                                     num_df=text_df,
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

            #test_env.render_all(path= "./plots/w2vsumRnnApproaches/{}_CV".format("_".join([str(x).replace("/","-") for x in list(dict(param).values())])) + "{}.png".format(i))

            # save generated data from model evaluation
            pd.DataFrame(test_env.history).to_excel("./results/{}_CV".format(run.name) + "{}.xlsx".format(i), index=False)

            wandb_dict = test_env._total_ratios
            wandb_dict["cumReturn"] = test_env._total_return
            wandb_dict = {k + '_CV': v for k, v in wandb_dict.items()}
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