from research import prepare_data
import numpy as np
import pandas as pd
from research import testing
import itertools
import matplotlib.pyplot as plt

class PairsTrading():

    def __init__(self, std_multiplier, rolling_window, tp, sl):
        self.std_multiplier = std_multiplier
        self.rolling_window = rolling_window
        self.tp = tp
        self.sl = sl 
# tej funkcji calculate_strategy możesz nie sprawdzać 
    def calculate_strategy(self, stock_1, stock_2):
 

        mean_price = (stock_1 - stock_2).rolling(window=self.rolling_window).mean()
        std_deviation = (stock_1 + stock_2).rolling(window=self.rolling_window).std()
        upper_bound = mean_price + (std_deviation * self.std_multiplier)
        lower_bound = mean_price - (std_deviation * self.std_multiplier)

        signals = pd.DataFrame(index=stock_1.index)
        signals['spread'] = (stock_1 - stock_2).abs()
        signals['mean'] = mean_price
        signals['upper'] = upper_bound 
        signals['lower'] = lower_bound
        signals['positions'] = 0

        signals['positions'] = np.where(signals['spread'] > signals['upper'], -1, 
                            np.where(signals['spread'] < signals['lower'], 1, 0))
        signals['positions'] = signals['positions'].replace(to_replace=0, method='ffill')

        entry_price = 0
        profit = []
        ticket_long = 0
        ticket_short = 0

        # Iteracja 
        for i, row in signals.iterrows():
            if ticket_long == 0 and row['positions'] == 1:
                entry_price = row['spread']
                ticket_long = 1
            elif ticket_long == 1 and row['spread'] <= row['mean']:
                profit.append(((row['spread'] - entry_price)/entry_price))
                ticket_long = 0

            if ticket_short == 0 and row['positions'] == -1:
                entry_price = row['spread']
                ticket_short = 1
            elif ticket_short == 1 and row['spread'] >= row['mean']:
                profit.append(((entry_price - row['spread'])/row['spread']))
                ticket_short = 0

        return pd.DataFrame(profit)



    def calculate_strategy_tp_sl(self, stock_1, stock_2):
        #wybieram średnią różnicę w cenach pomiedzy instrumentami z ostatnich x interwałów
        # wyliczam std i ustawiam jako threshold. Gdy przekroczy te poziomy to otwieram short/long z tp i sl
        mean_price = (stock_1 - stock_2).rolling(window=self.rolling_window).mean()
        std_deviation = (stock_1 - stock_2).rolling(window=self.rolling_window).std()
        upper_bound = mean_price + (std_deviation * self.std_multiplier)
        lower_bound = mean_price - (std_deviation * self.std_multiplier)

        signals = pd.DataFrame(index=stock_1.index)
        signals['spread'] = stock_1 - stock_2
        signals['mean'] = mean_price
        signals['upper'] = upper_bound
        signals['lower'] = lower_bound
        signals['positions'] = 0

        signals['positions'] = np.where(signals['spread'] > signals['upper'], -1, 
                            np.where(signals['spread'] < signals['lower'], 1, 0))
        signals['positions'] = signals['positions'].replace(to_replace=0, method='ffill')

        entry_price = 0
        profit = []
        ticket_long = 0
        ticket_short = 0

        tp_level = 0
        sl_level = 0
        signals = signals.dropna().reset_index()

        for i, row in signals.iterrows():

            # Obliczanie profitu włącznie z otwartymi pozycjami
            # w obliczaniu profitu na dole jest 'abs' bo czasami row['spread'] jest dodatni a czasami przechodzi w ujemny
            if i > 0:
                entry_price_yesterday = signals.at[i - 1, 'spread']
                if ticket_long == 1:
                    profit.append(((row['spread'] - entry_price_yesterday) / abs(entry_price_yesterday)) * 100)
                if ticket_short == 1:
                    profit.append(-((row['spread'] - entry_price_yesterday) / abs(entry_price_yesterday)) * 100)
            if i==0:
                entry_price_yesterday = row['spread']

            # Otwieranie i zarządzanie pozycjami długimi
            if ticket_long == 0 and ticket_short == 0 and row['positions'] == 1:
                entry_price = row['spread']
                ticket_long = 1
                tp_level = entry_price + abs(entry_price * self.tp)
                sl_level = entry_price - abs(entry_price * self.sl)
                profit.append(0)
                continue

            if ticket_long == 1:
                if row['spread'] >= tp_level or row['spread'] <= sl_level:
                    ticket_long = 0
                    tp_level = 0
                    sl_level = 0
                    continue

            # Otwieranie i zarządzanie pozycjami krótkimi
            if ticket_short == 0 and ticket_long == 0 and row['positions'] == -1:
                entry_price = row['spread']
                ticket_short = 1
                tp_level = entry_price - abs(entry_price * self.tp)
                sl_level = entry_price + abs(entry_price * self.sl)
                profit.append(0)
                continue

            if ticket_short == 1:
                if row['spread'] <= tp_level or row['spread'] >= sl_level:
                    ticket_short = 0
                    tp_level = 0
                    sl_level = 0
                    continue

            # Nie ma otwartych pozycji
            if ticket_long == 0 and ticket_short == 0:
                profit.append(0)

        return pd.DataFrame(profit)



    def execute_strategy(self, data, pairs):
        all_profits = pd.DataFrame()
    # zbieram wyniki z okresów testowych dla par kointegracyjnych
        for pair in pairs:
            stock_1 = data[pair[0]]
            stock_2 = data[pair[1]]
            profit_pair = self.calculate_strategy_tp_sl(stock_1, stock_2)
            all_profits = pd.concat([all_profits, profit_pair], ignore_index=True, axis=1)

        daily_returns = all_profits.sum(axis=1)
        cumulative = daily_returns.cumsum()
        #USUWAM 0 DO SREDNIEJ 
        daily_returns_clear = daily_returns[daily_returns!= 0]

        mean_return = daily_returns_clear.mean()
        std_dev = daily_returns_clear.std()

        sharpe = (mean_return / std_dev) * np.sqrt(252)

        print('symulacja zakończona')

        return cumulative, sharpe