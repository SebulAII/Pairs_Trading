from strategy import PairsTrading
from objective import objective
import optuna
from research import prepare_data
from research import find_cointegrated_pairs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    etf = 'XLE'
    n_trials = 250
    X_train, X_test = prepare_data(etf)
    cointegrated_pairs = find_cointegrated_pairs(X_train)

    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, X_train, cointegrated_pairs), n_trials=n_trials)

    best_params = study.best_params
    print("Best parameters:", best_params)

    strategy = PairsTrading(best_params['std_multiplier'], best_params['rolling_window'], best_params['tp'], best_params['sl'])
    test_results,sharpe = strategy.execute_strategy(X_test, cointegrated_pairs)
    
    plt.plot(test_results)
    plt.title(f'{etf} sharpe: {sharpe:.2f} profit%{test_results.iloc[-1]:.2f}std:{best_params["std_multiplier"]:.2f}, rolling:{best_params["rolling_window"]:.2f} tp: {best_params["tp"]:.2f}, sl: {best_params["sl"]:.2f}')
    plt.savefig(f'{etf}.png')