from tqdm import tqdm

from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

import gym
from environment import TextAssetTradingEnv
import numpy as np

from sklearn.model_selection import ParameterGrid, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from gym.envs.registration import register
register(
    id='textassettrading-v0',
    entry_point='environment:TextAssetTradingEnv',
)

# create an agent which can be trained on a dataset and then be used in an dynamic environment to predict the next action
class autoReg_agent:
    def __init__(self, number_assets, use_price= True, shared_model= True):
        self.number_assets = number_assets
        self.use_price = use_price
        self.shared_model = shared_model
        if self.shared_model == "shared":
            self.model = LinearRegression(normalize=True)
            self.enc = None
        elif self.shared_model == "sharedWithIndicator":
            self.model = LinearRegression(normalize=True)
            self.enc = OneHotEncoder(handle_unknown='ignore')
        else:
            self.model = [LinearRegression(normalize=True) for asset in range(number_assets)]
            self.enc = None

    def choose_action(self, obs):
        obs = obs[:, self.number_assets:]
        obs = np.expand_dims(obs.flatten(order="F"), axis= 0)

        # use different model dependent on if the data is shared between the assets or not
        if self.shared_model == "shared":
            X_temp = np.split(obs, obs.shape[1]/self.number_assets, axis= 1)
            X_temp = np.column_stack([X_1_temp.flatten()for X_1_temp in X_temp])
            X_temp[np.isnan(X_temp)] = 0
            if self.use_price:
                scores = self.model.predict(X_temp)
            else:
                scores = self.model.predict(X_temp[:, 1::2])
        elif self.shared_model == "sharedWithIndicator":
            X_cc = np.array(list(range(10)) * obs.shape[0])
            X_cc = self.enc.fit_transform(X_cc.reshape(-1, 1)).toarray()
            X_temp = np.split(obs, obs.shape[1] / self.number_assets, axis=1)
            X_temp = np.column_stack([X_1_temp.flatten() for X_1_temp in X_temp])
            X_temp[np.isnan(X_temp)] = 0
            X_temp = np.concatenate([X_temp, X_cc], axis=1)
            if self.use_price:
                scores = self.model.predict(X_temp)
            else:
                scores = self.model.predict(X_temp[:, 1::2])
        else:
            scores = np.array([])
            for i in range(self.number_assets):
                X_temp = obs[:, i::self.number_assets]
                X_temp[np.isnan(X_temp)] = 0
                if self.use_price:
                    scores = np.append(scores, self.model[i].predict(X_temp))
                else:
                    scores = np.append(scores, self.model[i].predict(X_temp[:, 1::2]))
        return scores.flatten()

    def remember(self, obs, act, reward, new_state, done):
        pass

    def learn(self):
        pass

    def train(self, X, y):
        # create different model dependent on if the data is shared between the assets or not
        if self.shared_model == "shared":
            X_temp = np.split(X, X.shape[1]/self.number_assets, axis= 1)
            X_temp = np.column_stack([X_1_temp.flatten()for X_1_temp in X_temp])
            X_temp[np.isnan(X_temp)] = 0
            y_temp = y.flatten()
            if self.use_price:
                self.model.fit(X_temp, y_temp)
            else:
                self.model.fit(X_temp[:, 1::2], y_temp)
        elif self.shared_model == "sharedWithIndicator":
            X_cc = np.array(list(range(10)) * X.shape[0])
            X_cc = self.enc.fit_transform(X_cc.reshape(-1, 1)).toarray()
            X_temp = np.split(X, X.shape[1] / self.number_assets, axis=1)
            X_temp = np.column_stack([X_1_temp.flatten() for X_1_temp in X_temp])
            X_temp[np.isnan(X_temp)] = 0
            X_temp = np.concatenate([X_temp, X_cc], axis=1)
            y_temp = y.flatten()
            if self.use_price:
                self.model.fit(X_temp, y_temp)
            else:
                self.model.fit(X_temp[:, 1::2], y_temp)
        else:
            for i in range(self.number_assets):
                X_temp = X[:, i::self.number_assets]
                X_temp[np.isnan(X_temp)] = 0
                y_temp = y[:, i]
                if self.use_price:
                    self.model[i].fit(X_temp, y_temp)
                else:
                    self.model[i].fit(X_temp[:, 1::2], y_temp)


param_grid = ParameterGrid({
    "feature": ["sentiment_bert", "sentiment_LM", "sentiment_Vader"],
    "window_size": [1, 2, 3, 4, 5, 6, 7],
    "source": ["Twitter", "CoinTelegraph"],
    "shared_model": ["seperated", "shared", "sharedWithIndicator"],
    "use_price": [True, False]
})

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

for param in tqdm(param_grid):
    # load Data
    num_df = pq.ParquetDataset("data/Content.parquet", validate_schema=False, filters=[('cc', 'in', ccs), ('source', '=', param["source"])])
    num_df = num_df.read(columns=[param["feature"]]).to_pandas()
    num_df["date"] = pd.to_datetime(num_df["date"]).apply(lambda x: x.date())
    num_df = num_df.set_index("date").drop("source", axis=1)

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

        # create gym environments
        train_env = gym.make('textassettrading-v0',
                            price_df=price_df,
                            window_size=param["window_size"],
                            num_df=num_df,
                            timeframe_bound=(train_start_date, test_roll_start_date),
                            reward=None,
                            max_texts_per_day=30,
                            max_words_per_text=20,
                            turbulence_threshold=np.inf,
                            trade_fee_bid_percent=0.005,
                            trade_fee_ask_percent=0.005,
                            force_trades=True,
                            provide_actions=False)

        train_X, train_y = train_env._get_dataframes()

        test_env = gym.make('textassettrading-v0',
                            price_df=price_df,
                            window_size=param["window_size"],
                            num_df=num_df,
                            timeframe_bound=(test_roll_start_date, test_roll_end_date),
                            reward=train_env.reward_type,
                            max_texts_per_day=train_env.texts_per_day,
                            max_words_per_text=train_env.words_per_text,
                            turbulence_threshold=np.inf,
                            trade_fee_bid_percent=train_env.trade_fee_ask_percent,
                            trade_fee_ask_percent=train_env.trade_fee_bid_percent,
                            force_trades=train_env.force_trades,
                            provide_actions=train_env.provide_actions)

        agent = autoReg_agent(test_env.stock_dim, use_price= param["use_price"], shared_model= param["shared_model"])

        agent.train(train_X, train_y)

        obs = test_env.reset()
        done = False
        score = 0
        m = 0

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

        #test_env.render_all(path= "./plots/autoregApproaches/{}_{}_{}_{}_{}_CV".format(*list(param.values()))+"{}.png".format(i))
        pd.DataFrame(test_env.history).to_excel("./results/autoregApproaches/{}_{}_{}_{}_{}_CV".format(*list(param.values())) + "{}.xlsx".format(i), index=False)

        for key in metrics.keys():
            if key == "cumReturn":
                metrics[key].append(test_env._total_return)
            else:
                metrics[key].append(test_env._total_ratios[key])

    save_dict = {}
    for key in metrics.keys():
        save_dict[key] = np.mean(metrics[key])
        save_dict[key + "_std"] = np.std(metrics[key])
        save_dict[key + "_q1"] = np.quantile(metrics[key], 0.25)
        save_dict[key + "_q3"] = np.quantile(metrics[key], 0.75)
        save_dict[key + "_min"] = np.max(metrics[key])
        save_dict[key + "_max"] = np.min(metrics[key])
        save_dict[key + "_med"] = np.median(metrics[key])

    save_dict = {**param, **save_dict}
    results_df = results_df.append(pd.Series(save_dict), ignore_index= True)

results_df.to_excel("./results/Overview_autoregApproaches.xlsx", index= False)