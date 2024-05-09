
import numpy as np
import pandas as pd
from strategy import PairsTrading

def objective(trial, X_train, cointegrated_pairs):
    std_multiplier = trial.suggest_float('std_multiplier', 1, 3, step = 0.001)
    rolling_window = trial.suggest_int('rolling_window', 20, 100)
    tp = trial.suggest_float('tp', 0.01, 0.15, step=0.001)
    sl = trial.suggest_float('sl', 0.01, 0.15, step = 0.001)

    strategy = PairsTrading(std_multiplier, rolling_window, tp, sl)
    # po puszczeniu optymalizacji na treningowym zapisuje najlepsze parametry i puszczam na testowym
    # na tych samych parach ktore byly kointegrowalne na okresie treningowym
    cumulative, sharpe  = strategy.execute_strategy(X_train, cointegrated_pairs)

    return (-1) * cumulative.iloc[-1]
    #return (-1) * sharpe