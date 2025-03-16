import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import time
import random

from functions import modelling as mdl
from functions import evaluation as eval

if __name__ == "__main__":

    GLOBAL_SEED = 42 # Claaaaaaasic

    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    
    df = pd.read_excel("data/Hourly_energy_c_2022-25.xlsx")
    df = df[['datetime_beginning_utc', 'mw']]
    df["Date"] = pd.to_datetime(df["datetime_beginning_utc"])
    df.set_index("Date", inplace=True)
    df.drop("datetime_beginning_utc", axis=1, inplace=True)
    
    print(f"Dataset loaded: {len(df)} points")

    test_start_date = pd.to_datetime('2025-01-01 05:00:00')

    train_data = df[df.index < test_start_date]
    test_data = df[df.index >= test_start_date].iloc[:40]
    
    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")

    # Parameters that I have done quiet a lot of messing with

    window_size = 512
    forecast_horizon = 1

    scattering_params = {
        'J': int(np.log2(window_size)),
        'Q': 8,
        'T': window_size,
        'shape': (window_size,),
    }

    lstm_params = {
        'hidden_dim': 512,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 25,
        'num_layers': 3,
    }

    print("Starting forecasting...")
    print(f"Global random seed: {GLOBAL_SEED}")
    start_time = time.time()
    lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, actual_values, arima_forecasts, training_times, lstm_scattering_model, pure_lstm_model, scattering_only_model = mdl.rolling_window_forecast_scattering_lstm(
        train_data, test_data, window_size, forecast_horizon, scattering_params, lstm_params, random_seed=GLOBAL_SEED
    )
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    metrics = eval.evaluate_forecasts(actual_values, lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, arima_forecasts)
    
    # A couple simple plots
    plt.figure(figsize=(12, 6))
    test_dates = test_data.index[:len(lstm_scattering_forecasts)]
    plt.plot(test_dates, actual_values, label='Actual Energy', color='blue', linewidth=2)
    plt.plot(test_dates, lstm_scattering_forecasts, label='LSTM+Wavelet Forecast', color='red', linestyle='--')
    plt.plot(test_dates, scattering_only_forecasts, label='Scattering-Only LSTM Forecast', color='orange', linestyle='--')
    plt.plot(test_dates, pure_lstm_forecasts, label='Pure LSTM Forecast', color='purple', linestyle=':')
    plt.plot(test_dates, arima_forecasts, label='ARIMA Forecast', color='green', linestyle='-.')
    plt.title('US Hourly Energy Forecasting', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Energy (MW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.figtext(0.15, 0.15,
        f'LSTM+Wavelet RMSE: {metrics["lstm_scattering"]["rmse"]:.2f}\n'
        f'Scattering-Only RMSE: {metrics["scattering_only"]["rmse"]:.2f}\n'
        f'Pure LSTM RMSE: {metrics["pure_lstm"]["rmse"]:.2f}\n'
        f'ARIMA RMSE: {metrics["arima"]["rmse"]:.2f}',
        bbox=dict(facecolor='white', alpha=0.8))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    mse_values = [metrics["lstm_scattering"]["mse"], metrics["scattering_only"]["mse"], 
                 metrics["pure_lstm"]["mse"], metrics["arima"]["mse"]]
    plt.bar(['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'ARIMA'], 
           mse_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Mean Squared Error')
    plt.ylabel('MSE')
    
    plt.subplot(2, 2, 2)
    mae_values = [metrics["lstm_scattering"]["mae"], metrics["scattering_only"]["mae"], 
                 metrics["pure_lstm"]["mae"], metrics["arima"]["mae"]]
    plt.bar(['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'ARIMA'], 
           mae_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    
    plt.subplot(2, 2, 3)
    rmse_values = [metrics["lstm_scattering"]["rmse"], metrics["scattering_only"]["rmse"], 
                  metrics["pure_lstm"]["rmse"], metrics["arima"]["rmse"]]
    plt.bar(['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'ARIMA'], 
           rmse_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Root Mean Squared Error')
    plt.ylabel('RMSE')
    
    plt.subplot(2, 2, 4)
    dir_acc_values = [metrics["lstm_scattering"]["dir_acc"], metrics["scattering_only"]["dir_acc"], 
                     metrics["pure_lstm"]["dir_acc"], metrics["arima"]["dir_acc"]]
    plt.bar(['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'ARIMA'], 
           dir_acc_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Directional Accuracy')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    avg_training_times = [
        np.mean(training_times['lstm_scattering']),
        np.mean(training_times['scattering_only']),
        np.mean(training_times['pure_lstm']),
        np.mean(training_times['arima'])
    ]
    plt.bar(['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'ARIMA'], 
           avg_training_times, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Average Training Times')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    plt.show()