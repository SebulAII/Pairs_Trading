
import numpy as np
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.vector_ar.vecm as vecm
from arch.unitroot import engle_granger
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from hurst import compute_Hc

def prepare_data(etf, split_ratio = 0.8):
    
    # etf = 'XLE'
    folder_path = f'/home/swozniczka/Dokumenty/AII/Pairs_Trading/Dane/Equities/{etf}'

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    dfs = {}
    first_dates = {}

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path, usecols=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.rename(columns={'Close': os.path.splitext(file_name)[0]}, inplace=True)
        dfs[os.path.splitext(file_name)[0]] = df
        first_dates[os.path.splitext(file_name)[0]] = df.index[0]

    sorted_first_dates_dict = dict(sorted(first_dates.items(), key=lambda x: x[1]))

    # Obliczenie 10% liczby spółek do usunięcia WAZNE!!!!!!
    num_to_remove = int(len(sorted_first_dates_dict) * 0.1)

    # Usunięcie najmłodszych spółek (10%)
    for _ in range(num_to_remove):
        sorted_first_dates_dict.popitem()

    # Wybór najstarszych 90% spółek
    selected_companies = sorted_first_dates_dict.keys()

    combined_df = pd.concat([dfs[company] for company in selected_companies], axis=1, join='outer')

    combined_df = combined_df.dropna()


    # Skalowanie danych
    scaler = MaxAbsScaler()
    combined_df_scaled = pd.DataFrame(scaler.fit(combined_df).transform(combined_df), columns=combined_df.columns)

## podział train/test
    split_index = int(len(combined_df_scaled) * split_ratio)
    X_train = combined_df_scaled[:split_index]
    X_test = combined_df_scaled[split_index:]
#dostajemy df z okresem treningowym i testowym z zeskalowanymi cenami
    return X_train, X_test

def testing(stock_1, stock_2):

    #krytyczne wartosci
    critical_values = {0: {.9: 13.4294, .95: 15.4943, .99: 19.9349},
                    1: {.9: 2.7055, .95: 3.8415, .99: 6.6349}}
    trace0_cv = critical_values[0][.95]
    trace1_cv = critical_values[1][.95]

    # Johansen test
    df_pair = pd.concat([stock_1, stock_2], axis=1)
    var = vecm.VAR(df_pair.values)
    lags = var.select_order().aic
    johansen_test = vecm.coint_johansen(df_pair, 0, lags)
    johansen_result = (johansen_test.lr1[0] > trace0_cv and johansen_test.lr2[0] > trace1_cv)
    
    # Engle-Granger test
    eg_test = engle_granger(stock_1, stock_2, trend="c")
    eg_result = eg_test.pvalue < 0.05

    # hurst
    epsilon = 1e-30
    diff = abs(stock_1 - stock_2) + epsilon

    Hurst, C, diff = compute_Hc(diff, kind='price', simplified=True)

# zwracam wyniki testów na kointegrację oraz na Hursta
    return johansen_result, eg_result, Hurst

def find_cointegrated_pairs(data):
    #przepuszczam testy kointegracyjne z funkcji testing i zwracam 
    # pary ktore sa kointegrowalne na okresie treningowym (przeszly oba testy)
    train_data = data
    cointegrated_pairs = []
    for pair in itertools.combinations(train_data.columns, 2):
        if testing(train_data[pair[0]], train_data[pair[1]]):
            johansen, eg_restults, hurst = testing(train_data[pair[0]], train_data[pair[1]]) 
            if johansen and eg_restults:
                cointegrated_pairs.append(pair)
    return cointegrated_pairs