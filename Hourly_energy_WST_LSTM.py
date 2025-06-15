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
    mdl.set_random_seeds(GLOBAL_SEED)
    
    df = pd.read_excel("/Users/jakubfridrich/Documents/Hourly_energy_c_COMED_2005-25.xlsx")
    df = df[['datetime_beginning_ept', 'mw']]
    df["Date"] = pd.to_datetime(df["datetime_beginning_ept"])
    df.set_index("Date", inplace=True)
    df.drop("datetime_beginning_ept", axis=1, inplace=True)

    print(f"Dataset loaded: {len(df)} points")

    test_start_date = pd.to_datetime('2022-04-01 00:00:00')

    train_data = df[df.index < test_start_date]
    test_data = df[df.index >= test_start_date]

    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")

    # Parameters that I have done quiet a lot of messing with

    window_size = 256
    forecast_horizon = 1
    time_lags = 30
    step_size = 5
    energy_threshold = 0.99
    
    # Multi-step forecasting parameters
    multi_step_forecast = True
    FORECAST_STEPS = 772
    forecast_steps = 772
    NUMBER_OF_TRIALS_RANDOM_SEARCH = 7
    num_random_starts = 20
    
    scattering_params = {
        'J': int(np.log2(window_size)),
        'Q': 8,
        'T': window_size,
        'shape': (window_size,),
    }
    
    lstm_params_grid = {
        'hidden_dims': [20, 40, 60],
        'dropout_rates': [0.2, 0.3, 0.4],
        'learning_rates': [0.001, 0.003, 0.005],
        'num_layers': 1,
        'full_epochs': 150
    }

    print("Starting forecasting with time-series cross validation and random starting points...")
    print(f"Global random seed: {GLOBAL_SEED}")
    print(f"Number of random starting points: {num_random_starts}")
    start_time = time.time()

    (all_lstm_scattering_sequences, all_pure_lstm_sequences, all_scattering_only_sequences, 
     all_actual_sequences, all_sarima_sequences, training_times, lstm_scattering_model, 
     pure_lstm_model, scattering_only_model, random_start_indices, 
     best_config) = mdl.rolling_window_forecast_scattering_lstm_cv(
        train_data, test_data, window_size, forecast_horizon, scattering_params, lstm_params_grid, 
        time_lags=time_lags, random_seed=GLOBAL_SEED, step_size=step_size, energy_threshold=energy_threshold,
        multi_step=multi_step_forecast, forecast_steps=forecast_steps, n_splits=3, 
        num_random_starts=num_random_starts
    )

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    all_metrics = []
    for i in range(len(all_actual_sequences)):
        metrics = eval.evaluate_forecasts(
            all_actual_sequences[i], 
            all_lstm_scattering_sequences[i], 
            all_pure_lstm_sequences[i], 
            all_scattering_only_sequences[i], 
            all_sarima_sequences[i]
        )
        all_metrics.append(metrics)

    avg_metrics = {
        'lstm_scattering': {},
        'scattering_only': {},
        'pure_lstm': {},
        'sarima': {}
    }

    for model in avg_metrics.keys():
        for metric in ['mse', 'rmse', 'mae', 'mape', 'dir_acc']:
            values = [m[model][metric] for m in all_metrics]
            avg_metrics[model][metric] = np.mean(values)
            avg_metrics[model][f'{metric}_std'] = np.std(values)

    print(f"\n===== Averaged Metrics Across {num_random_starts} Starting Points =====")
    for model in ['lstm_scattering', 'scattering_only', 'pure_lstm', 'sarima']:
        model_name = {
            'lstm_scattering': 'LSTM+Scattering',
            'scattering_only': 'Scattering-Only LSTM', 
            'pure_lstm': 'Pure LSTM',
            'sarima': 'SARIMA'
        }[model]
        
        print(f"\n{model_name} Metrics (Mean ± Std):")
        print(f"  MSE: {avg_metrics[model]['mse']:.2f} ± {avg_metrics[model]['mse_std']:.2f}")
        print(f"  RMSE: {avg_metrics[model]['rmse']:.2f} ± {avg_metrics[model]['rmse_std']:.2f}")
        print(f"  MAE: {avg_metrics[model]['mae']:.2f} ± {avg_metrics[model]['mae_std']:.2f}")
        print(f"  MAPE: {avg_metrics[model]['mape']:.2f}% ± {avg_metrics[model]['mape_std']:.2f}%")
        print(f"  Directional Accuracy: {avg_metrics[model]['dir_acc']:.2f}% ± {avg_metrics[model]['dir_acc_std']:.2f}%")

    lstm_scattering_forecasts = np.mean(all_lstm_scattering_sequences, axis=0)
    pure_lstm_forecasts = np.mean(all_pure_lstm_sequences, axis=0)
    scattering_only_forecasts = np.mean(all_scattering_only_sequences, axis=0)
    sarima_forecasts = np.mean(all_sarima_sequences, axis=0)
    actual_values = np.mean(all_actual_sequences, axis=0)

    # This for-loop generates four sample forccasts 

    sample_indices = np.random.choice(range(len(random_start_indices)), size=min(4, len(random_start_indices)), replace=False)
    sample_starts = [random_start_indices[i] for i in sample_indices]

    plt.figure(figsize=(20, 16))

    for idx, start_point in enumerate(sample_starts):
        plt.subplot(2, 2, idx + 1)
        
        context_window = 48
        start_idx = max(0, start_point - context_window)
        end_idx = min(len(test_data), start_point + context_window)
        
        window_dates = test_data.index[start_idx:end_idx]
        window_actual = test_data.iloc[start_idx:end_idx].values.flatten()
        
        plt.plot(window_dates, window_actual, 'b-', linewidth=2, label='Actual Energy', alpha=0.8)
        
        forecast_date = test_data.index[start_point + forecast_horizon - 1]
        actual_value = all_actual_values[np.where(np.array(random_start_indices) == start_point)[0][0]]
        lstm_scattering_pred = all_lstm_scattering_forecasts[np.where(np.array(random_start_indices) == start_point)[0][0]]
        scattering_only_pred = all_scattering_only_forecasts[np.where(np.array(random_start_indices) == start_point)[0][0]]
        pure_lstm_pred = all_pure_lstm_forecasts[np.where(np.array(random_start_indices) == start_point)[0][0]]
        sarima_pred = all_sarima_forecasts[np.where(np.array(random_start_indices) == start_point)[0][0]]
        
        plt.scatter([forecast_date], [actual_value], color='blue', s=100, marker='o', label='Actual Target', zorder=5)
        plt.scatter([forecast_date], [lstm_scattering_pred], color='red', s=100, marker='^', label='LSTM+Wavelet', zorder=5)
        plt.scatter([forecast_date], [scattering_only_pred], color='orange', s=100, marker='s', label='Scattering-Only', zorder=5)
        plt.scatter([forecast_date], [pure_lstm_pred], color='purple', s=100, marker='d', label='Pure LSTM', zorder=5)
        plt.scatter([forecast_date], [sarima_pred], color='green', s=100, marker='v', label='SARIMA', zorder=5)
        
        plt.axvline(x=forecast_date, color='gray', linestyle='--', alpha=0.5, label='Forecast Point')
        
        plt.title(f'Sample Forecast {idx+1}\nStarting Point: {start_point}, Date: {forecast_date.strftime("%Y-%m-%d %H:%M")}', fontsize=12)
        plt.xlabel('Date')
        plt.ylabel('Energy (MW)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        lstm_error = abs(actual_value - lstm_scattering_pred)
        scattering_error = abs(actual_value - scattering_only_pred)
        pure_error = abs(actual_value - pure_lstm_pred)
        sarima_error = abs(actual_value - sarima_pred)
        
        plt.text(0.02, 0.98, f'Errors:\nLSTM+Wavelet: {lstm_error:.1f}\nScattering-Only: {scattering_error:.1f}\nPure LSTM: {pure_error:.1f}\nSARIMA: {sarima_error:.1f}', 
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('sample_forecast_comparisons.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Overall forecasting summary is here, simpler plots below

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    sample_indices = range(len(actual_values))
    plt.scatter(sample_indices, actual_values, label='Actual Energy', color='blue', alpha=0.7, s=40)
    plt.scatter(sample_indices, lstm_scattering_forecasts, label='LSTM+Wavelet', color='red', alpha=0.7, s=40)
    plt.scatter(sample_indices, scattering_only_forecasts, label='Scattering-Only', color='orange', alpha=0.7, s=40)
    plt.scatter(sample_indices, pure_lstm_forecasts, label='Pure LSTM', color='purple', alpha=0.7, s=40)
    plt.scatter(sample_indices, sarima_forecasts, label='SARIMA', color='green', alpha=0.7, s=40)
    plt.title(f'Forecast Results Summary\n({num_random_starts} Random Starting Points)', fontsize=14)
    plt.xlabel('Random Sample Index', fontsize=12)
    plt.ylabel('Energy (MW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    lstm_scattering_errors = np.abs(actual_values - lstm_scattering_forecasts)
    scattering_only_errors = np.abs(actual_values - scattering_only_forecasts)
    pure_lstm_errors = np.abs(actual_values - pure_lstm_forecasts)
    sarima_errors = np.abs(actual_values - sarima_forecasts)

    plt.boxplot([lstm_scattering_errors, scattering_only_errors, pure_lstm_errors, sarima_errors], 
                labels=['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'SARIMA'])
    plt.title('Absolute Forecast Errors Distribution', fontsize=14)
    plt.ylabel('Absolute Error (MW)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.scatter(actual_values, lstm_scattering_forecasts, alpha=0.7, label='LSTM+Wavelet', color='red', s=50)
    plt.scatter(actual_values, scattering_only_forecasts, alpha=0.7, label='Scattering-Only', color='orange', s=50)
    plt.scatter(actual_values, pure_lstm_forecasts, alpha=0.7, label='Pure LSTM', color='purple', s=50)
    plt.scatter(actual_values, sarima_forecasts, alpha=0.7, label='SARIMA', color='green', s=50)

    min_val = min(np.min(actual_values), np.min(lstm_scattering_forecasts))
    max_val = max(np.max(actual_values), np.max(lstm_scattering_forecasts))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Prediction')

    plt.title('Predictions vs Actual Values', fontsize=14)
    plt.xlabel('Actual Energy (MW)', fontsize=12)
    plt.ylabel('Predicted Energy (MW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    model_names = ['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'SARIMA']
    mse_values = [metrics["lstm_scattering"]["mse"], metrics["scattering_only"]["mse"], 
                    metrics["pure_lstm"]["mse"], metrics["sarima"]["mse"]]

    bars = plt.bar(model_names, mse_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Mean Squared Error Comparison', fontsize=14)
    plt.ylabel('MSE', fontsize=12)
    plt.xticks(rotation=45)

    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 3, 5)
    mae_values = [metrics["lstm_scattering"]["mae"], metrics["scattering_only"]["mae"], 
                    metrics["pure_lstm"]["mae"], metrics["sarima"]["mae"]]
    bars = plt.bar(model_names, mae_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Mean Absolute Error Comparison', fontsize=14)
    plt.ylabel('MAE', fontsize=12)
    plt.xticks(rotation=45)

    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.02, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 3, 6)
    dir_acc_values = [metrics["lstm_scattering"]["dir_acc"], metrics["scattering_only"]["dir_acc"], 
                        metrics["pure_lstm"]["dir_acc"], metrics["sarima"]["dir_acc"]]
    bars = plt.bar(model_names, dir_acc_values, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Directional Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)

    for bar, value in zip(bars, dir_acc_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dir_acc_values)*0.02, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comprehensive_forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    avg_training_times = [
        np.mean(training_times['lstm_scattering']),
        np.mean(training_times['scattering_only']),
        np.mean(training_times['pure_lstm']),
        np.mean(training_times['sarima'])
    ]
    bars = plt.bar(['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'SARIMA'], 
            avg_training_times, color=['#e74c3c', '#f39c12', '#8e44ad', '#27ae60'])
    plt.title('Average Training Times', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, avg_training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_training_times)*0.02, 
                f'{value:.1f}s', ha='center', va='bottom', fontsize=11)

    plt.subplot(1, 2, 2)
    model_names = ['LSTM+Wavelet', 'Scattering-Only', 'Pure LSTM', 'SARIMA']
    mse_values = [metrics["lstm_scattering"]["mse"], metrics["scattering_only"]["mse"], 
                    metrics["pure_lstm"]["mse"], metrics["sarima"]["mse"]]
    colors = ['#e74c3c', '#f39c12', '#8e44ad', '#27ae60']

    plt.scatter(avg_training_times, mse_values, c=colors, s=150, alpha=0.8)

    for i, (name, time, mse) in enumerate(zip(model_names, avg_training_times, mse_values)):
        plt.annotate(name, (time, mse), xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, alpha=0.8)

    plt.title('Performance vs Training Time Trade-off', fontsize=14)
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('MSE (lower is better)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_and_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    model_mse = {
        'LSTM+Wavelet': metrics["lstm_scattering"]["mse"],
        'Scattering-Only': metrics["scattering_only"]["mse"], 
        'Pure LSTM': metrics["pure_lstm"]["mse"],
        'SARIMA': metrics["sarima"]["mse"]
    }

    best_model = min(model_mse, key=model_mse.get)
    print(f"{best_model} (MSE: {model_mse[best_model]:.2f})")