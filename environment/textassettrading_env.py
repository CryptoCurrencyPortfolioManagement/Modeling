import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from datetime import timedelta
from .utilities import *
from empyrical import cum_returns, sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio
import matplotlib.pyplot as plt


class TextAssetTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price_df, window_size, num_df= pd.DataFrame(), text_df= pd.DataFrame(), timeframe_bound= (None, None), reward=None, max_texts_per_day= 30, max_words_per_text= 20, turbulence_threshold= np.inf, trade_fee_bid_percent= 0.01, trade_fee_ask_percent= 0.01, force_trades= True, provide_actions= True):
        assert len(timeframe_bound) == 2, "timeframe bound does not have correct dimension"
        assert price_df.shape[0] > window_size, "timeframe smaller than window_size"
        assert reward == None or reward in ["sharpeRatio", "sortinoRatio", "calmarRatio", "mdd"], "choose a valid reward"

        if not timeframe_bound[0]:
            timeframe_bound[0] = price_df.index.min()
        if not timeframe_bound[1]:
            timeframe_bound[1] = price_df.index.max()

        self.price_df =  price_df.loc[(price_df.index.to_numpy() >= timeframe_bound[0]) & (price_df.index.to_numpy() <= timeframe_bound[1])]
        self.stock_dim = self.price_df.shape[1]

        self.window_size = window_size
        self.texts_per_day = max_texts_per_day
        self.words_per_text = max_words_per_text
        self.turbulence_threshold = turbulence_threshold
        self.provide_actions = provide_actions
        self.force_trades = force_trades
        self.reward_type = reward
        self.labels = list(price_df.columns)
        self.time_dim = None

        self.text_df = text_df
        self.num_df = num_df

        # handle empty dataframes
        if not self.text_df.empty:
            self.text_df = text_df.loc[(text_df.index.to_numpy() >= timeframe_bound[0]) & (text_df.index.to_numpy() <= timeframe_bound[1])]
        else:
            self.text_df = pd.DataFrame({self.text_df.columns[0]: [[np.nan] * (self.texts_per_day * self.words_per_text)] * self.stock_dim, self.text_df.columns[-1]: self.price_df.columns}, index= [timeframe_bound[0]]* self.stock_dim)
        if not self.num_df.empty:
            self.num_df = num_df.loc[(num_df.index.to_numpy() >= timeframe_bound[0]) & (num_df.index.to_numpy() <= timeframe_bound[1])]
        else:
            self.num_df = pd.DataFrame({**dict(zip(self.num_df.columns[:-1], [np.nan] * self.stock_dim)), **{self.num_df.columns[-1]: self.price_df.columns}}, index= [timeframe_bound[0]]* self.stock_dim)

        if not self.text_df.empty and not self.num_df.empty:
            self.observation_shape = (self.window_size,
                          self.stock_dim * 2  # last stock prices and last stock holdings
                          + self.words_per_text * self.texts_per_day * self.stock_dim  # concatenated text features
                          + (self.num_df.shape[1]-1) * self.stock_dim # concatenated sentiment features
                          )
        elif not self.text_df.empty:
            self.observation_shape = (self.window_size,
                          self.stock_dim * 2 # last stock prices and last stock holdings
                          + self.words_per_text * self.texts_per_day * self.stock_dim # concatenated text features
                          )
        elif not self.num_df.empty:
            self.observation_shape = (self.window_size,
                          self.stock_dim * 2  # last stock prices and last stock holdings
                          + (self.num_df.shape[1]-1) * self.stock_dim # concatenated sentiment features
                          )
        else:
            self.observation_shape = (window_size,
                                      self.stock_dim * 2)

        self.observation_space_descriptor = {}

        self.trade_fee_bid_percent = trade_fee_bid_percent # unit
        self.trade_fee_ask_percent = trade_fee_ask_percent # unit

        # data preprocessing
        self.prices, self.signal = self._process_data()

        # spaces
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.stock_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)


        # episode
        self._start_tick = self.window_size
        self._end_tick = self.prices.shape[0] - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._positions = None
        self._position_history = None
        self._return = None
        self._return_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._positions = np.zeros(self.stock_dim)
        self._return = 0.
        self._position_history = np.zeros((self.window_size, self.stock_dim))
        self._return_history = np.zeros(self.window_size)
        self._reward = 0.
        self._total_return = 0.
        self._total_return_history = np.zeros(self.window_size)
        self._total_ratios  = {
            "sharpeRatio": 0.,
            "sortinoRatio": 0.,
            "calmarRatio": 0.,
            "mdd": 0.,
        }
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    # simulate one step in the environment
    def step(self, actions):
        old_actions = actions
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        if self.force_trades:
            actions = actions - np.mean(actions)
        else:
            actions[actions < 0] = actions[actions < 0] / (np.abs(np.sum(actions[actions < 0])) / np.sum(actions[actions > 0]))

        # normalize actions so that all positive actions sum to 1 and all negative sum to -1
        actions[abs(actions) < 0.00001] = 0
        actions[actions < 0] = actions[actions < 0] / np.abs(np.sum(actions[actions < 0]))
        actions[actions > 0] = actions[actions > 0] / np.abs(np.sum(actions[actions > 0]))

        # check to see if all actions sum to 0
        if np.abs(np.sum(actions)) > 0.00001:
            print("error in creating cumulativly neutral positions ({})".format(actions))
            print("setting actions to zero")
            actions = np.zeros(actions.shape)

        if not self.provide_actions:
            actions = actions - self._positions

        value_old = np.sum(actions)

        # implement transaction costs
        actions[actions < 0] = actions[actions < 0] * (1 + self.trade_fee_bid_percent)
        actions[actions > 0] = actions[actions > 0] * (1 - self.trade_fee_ask_percent)

        value_new = np.sum(actions)

        if value_old < value_new:
            print("error in calculating the trading costs (old: {} - new: {})".format(value_old, value_new))

        #if not self.provide_actions:
        #    self._positions = actions
        #else:
        #    self._positions += actions

        old_positions = self._positions
        self._positions += actions

        #self._positions = self._positions / np.max(self._positions)

        if np.sum(self._positions) > 0.00001 or np.sum(self._positions) < -self.trade_fee_ask_percent*8:
            print("error in calculating the trading costs (posSum: {})".format(np.sum(self._positions)))

        if self.force_trades or np.abs(np.sum(actions[actions < 0])) != 0:
            self._last_trade_tick = self._current_tick

        self._update_return()
        self._update_ratios()

        self._calculate_reward()

        self._position_history = np.append(self._position_history, [self._positions], axis=0)

        observation = self._get_observation()

        info = {**{"date": self.time_dim[self._current_tick], "return": self._return, "total_return": self._total_return, "positions": self._positions.copy()}, **self._total_ratios}

        self._update_history(info)

        return observation, self._reward, self._done, info


    # load the data and bring it into the correct format
    def _process_data(self):
        self.price_df = self.price_df.sort_index(ascending=True)

        self.price_df[:] = self.price_df.values / self.price_df.shift(periods=1).values - 1
        prices = self.price_df.iloc[2:, :]

        self.time_dim = prices.index.to_numpy()

        self.observation_space_descriptor["price_data_dim"] = (1, prices.shape[1])
        signal = self.price_df.iloc[1:-1, :]
        signal.columns = signal.columns.to_list()

        if not self.text_df.empty:
            self.observation_space_descriptor["text_data_dim"] = (self.texts_per_day * self.words_per_text, len(self.labels))
            tqdm.pandas(desc="Process Text")
            temp_text = self.text_df.groupby(self.text_df.index).progress_apply(lambda x: generate_texts_per_day(x, self.texts_per_day, self.words_per_text, assets= self.labels))
            temp_text.index = pd.to_datetime(temp_text.index).date
            for date in tqdm(list(set(signal.index)-set(temp_text.index)), desc="Text fill Empty Slots"):
                temp_text = temp_text.append(pd.Series(temp_text.shape[1]*[[np.nan] * (self.texts_per_day * self.words_per_text)], index= temp_text.columns, name= date))
            temp_text = temp_text.applymap(lambda x: np.array(x))
            signal = pd.merge(signal, temp_text, how="left", left_index=True, right_index=True)

        if not self.num_df.empty:
            self.observation_space_descriptor["num_data_dim"] = (self.num_df.shape[1]-1, len(self.labels))
            tqdm.pandas(desc="Process Number")
            temp_num = generate_nums_per_day(self.num_df, assets= self.labels)
            for date in tqdm(list(set(signal.index) - set(temp_num.index)), desc="Number fill Empty Slots"):
                temp_num = temp_num.append(pd.Series(temp_num.shape[1] * [np.nan], index= temp_num.columns, name=date))
            signal = pd.merge(signal, temp_num, how="left", left_index= True, right_index= True)

        signal = signal.sort_index(ascending=True)

        assert prices.shape[0] == signal.shape[0], "mismatch in price and signal shape"

        signal = signal.apply(lambda x: pd.Series(np.concatenate([y if isinstance(y, (np.ndarray, np.generic)) else np.array([y]) for y in x.to_list()])), axis= 1)

        return prices.to_numpy(), signal.to_numpy()

    def _get_observation(self):
        obs_1 = self._position_history[(self._current_tick-self.window_size):self._current_tick]
        obs_2 = self.signal[(self._current_tick-self.window_size):self._current_tick]
        return np.concatenate((obs_1, obs_2), axis= 1)

    def _get_dataframes(self):
        prices = self.prices[self.window_size:]
        signal_with_lag = self.signal[self.window_size:]
        for window in range(self.window_size-1):
            signal_with_lag = np.append(signal_with_lag, np.roll(self.signal, window+1, axis=0)[self.window_size:], axis=1)
        return signal_with_lag, prices

    def _update_return(self):
        self._return = np.dot(self._positions,self.prices[self._current_tick])
        self._return_history = np.append(self._return_history, self._return)
        self._total_return = cum_returns(self._return_history[self.window_size:])[-1]
        self._total_return_history = np.append(self._total_return_history, self._total_return)

    def _update_ratios(self):
        self._total_ratios["sharpeRatio"] = sharpe_ratio(self._return_history[self.window_size:])
        self._total_ratios["sortinoRatio"] = sortino_ratio(self._return_history[self.window_size:])
        self._total_ratios["calmarRatio"] = calmar_ratio(self._return_history[self.window_size:])
        self._total_ratios["mdd"] = max_drawdown(self._return_history[self.window_size:])

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _calculate_reward(self):
        if self.reward_type:
            self._reward = self._total_ratios[self.reward_type] - self._reward
        else:
            self._reward = self._return

    # plot positions and returns of the current environment over time until the end tick
    def render(self, mode='human', close=False, path= None):
        def _plot_position(ax, positions):
            color = ["green" if x > 0 else "red" for x in positions]
            ax.bar(self.labels, positions, color=color)
            ax.set_xticklabels(ax.get_xticks(), rotation=90)
            ax.title.set_text('Positions')
        def _plot_cumreturns(ax, returns, max_len):
            plot_data = np.empty(max_len)
            plot_data[:] = np.nan
            plot_data[-1] = 0
            ax.plot(returns)
            ax.plot(plot_data, color= "white")
            ax.set_ylim([-1.05, 3.05])
            ax.title.set_text("Cumulative Return")

        fig, (ax1, ax2) = plt.subplots(2, 1)
        _plot_position(ax1, self._positions)
        _plot_cumreturns(ax2, self._total_return_history[self.window_size:], self._end_tick - self.window_size)

        fig.suptitle(
            "Date: {} \n".format(self.time_dim[self._current_tick]) +
            "Total Reward: %.6f" % self._total_return + ' ~ ' +
            "Total Sharpe: %.6f" % self._total_ratios["sharpeRatio"]
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        if path:
            plt.savefig(path)
        plt.pause(0.01)

    # plot positions and returns of the current environment over complete time
    def render_all(self, mode='human', path= None):
        def _plot_position(ax, positions):
            for i, label in enumerate(self.labels):
                ax.plot(positions[:, i], label=label)
            ax.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            ax.title.set_text('Positions')
        def _plot_cumreturns(ax, returns):
            ax.plot(returns, label= "Cumulative Returns")
            ax.title.set_text("Cumulative Returns")
            ax.set_ylim([-1.05, 1.05])

        fig, (ax1, ax2) = plt.subplots(2, 1)
        _plot_position(ax1, self._position_history[self.window_size:])
        _plot_cumreturns(ax2, self._total_return_history[self.window_size:])

        fig.suptitle(
            "Data {} to {} \n".format(self.time_dim[self.window_size+1], self.time_dim[-1]) +
            "Total Reward: %.6f" % self._total_return + ' ~ ' +
            "Total Sharpe: %.6f" % self._total_ratios["sharpeRatio"]
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        if path:
            plt.savefig(path)

        plt.pause(0.01)