import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import time
import random

from functions import modelling as mdl
from functions import evaluation as eval
from functions import data_processing

if __name__ == "__main__":
    
    GLOBAL_SEED = 42 # Claaaaaaasic
    mdl.set_random_seeds(GLOBAL_SEED)
    
    df = pd.read_excel("data/Hourly_energy_c_BC_2010-25.xlsx")
    df = df[['datetime_beginning_ept', 'mw']]
    df["Date"] = pd.to_datetime(df["datetime_beginning_ept"])
    df.set_index("Date", inplace=True)
    df.drop("datetime_beginning_ept", axis=1, inplace=True)

    print(f"Dataset loaded: {len(df)} points")

    test_start_date = pd.to_datetime('2023-11-30 03:00:00')

    train_data = df[df.index < test_start_date]
    test_data = df[df.index >= test_start_date]

    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")

    # (Hyper)parameters that I have done quite a lot of messing with

    WINDOW_SIZE = 128
    FORECAST_HORIZON = 1
    TIME_LAGS = 60
    STEP_SIZE = 7
    ENERGY_THRESHOLD = 0.99
    
    MULTI_STEP_FORECAST = True
    FORECAST_STEPS = 730
    NUMBER_OF_TRIALS_RANDOM_SEARCH = 20
    CV_SPLITS= 3
    NUM_RANDOM_STARTS= 20
    BATCH_SIZE = 32
    SCATTERING_SEQUENCE_LENGTH = 6
    
    scattering_params = {
        'J': int(np.log2(WINDOW_SIZE)),
        'Q': 4,
        'T': WINDOW_SIZE,
        'shape': (WINDOW_SIZE,),
    }
    
    lstm_params_grid = {
        'hidden_dims': [80, 100, 120],
        'dropout_rates': [0.1, 0.2, 0.3],
        'learning_rates': [0.001, 0.003, 0.005],
        'num_layers': [1, 2, 3],
        'full_epochs': 100
    }

    print("Starting forecasting with time-series cross validation and random starting points...")
    print(f"Global random seed: {GLOBAL_SEED}")
    print(f"Number of random starting points: {NUM_RANDOM_STARTS}")
    start_time = time.time()

    (all_lstm_scattering_sequences, all_pure_lstm_sequences, 
     all_actual_sequences, all_sarima_sequences, training_times, lstm_scattering_model, 
     pure_lstm_model, random_start_indices, 
     best_config) = mdl.rolling_window_forecast_scattering_lstm_cv(
        train_data, test_data, WINDOW_SIZE, FORECAST_HORIZON, scattering_params, lstm_params_grid, 
        time_lags=TIME_LAGS, random_seed=GLOBAL_SEED, step_size=STEP_SIZE, energy_threshold=ENERGY_THRESHOLD,
        multi_step=MULTI_STEP_FORECAST, forecast_steps=FORECAST_STEPS, n_splits=CV_SPLITS, 
        num_random_starts=NUM_RANDOM_STARTS, random_search_trials=NUMBER_OF_TRIALS_RANDOM_SEARCH, batch_size=BATCH_SIZE,
        scattering_sequence_length=SCATTERING_SEQUENCE_LENGTH
    )

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    all_metrics = []
    for i in range(len(all_actual_sequences)):
        metrics = eval.evaluate_forecasts(
            all_actual_sequences[i], 
            all_lstm_scattering_sequences[i], 
            all_pure_lstm_sequences[i], 
            all_sarima_sequences[i]
        )
        all_metrics.append(metrics)

    avg_metrics = {
        'lstm_scattering': {},
        'pure_lstm': {},
        'sarima': {}
    }

    for model in avg_metrics.keys():
        for metric in ['mse', 'rmse', 'mae', 'mape', 'dir_acc']:
            values = [m[model][metric] for m in all_metrics]
            avg_metrics[model][metric] = np.mean(values)
            avg_metrics[model][f'{metric}_std'] = np.std(values)

    print(f"\n======= Averaged metrics across {NUM_RANDOM_STARTS} starting points =====")
    for model in ['lstm_scattering', 'pure_lstm', 'sarima']:
        model_name = {
            'lstm_scattering': 'LSTM+Scattering',
            'pure_lstm': 'Pure LSTM',
            'sarima': 'SARIMA'
        }[model]
        
        print(f"\n{model_name} Metrics (Mean +- Std):")
        print(f"  MSE: {avg_metrics[model]['mse']:.2f} +- {avg_metrics[model]['mse_std']:.2f}")
        print(f"  RMSE: {avg_metrics[model]['rmse']:.2f} +- {avg_metrics[model]['rmse_std']:.2f}")
        print(f"  MAE: {avg_metrics[model]['mae']:.2f} +- {avg_metrics[model]['mae_std']:.2f}")
        print(f"  MAPE: {avg_metrics[model]['mape']:.2f}% +- {avg_metrics[model]['mape_std']:.2f}%")
        print(f"  Directional Accuracy: {avg_metrics[model]['dir_acc']:.2f}% +- {avg_metrics[model]['dir_acc_std']:.2f}%")

    lstm_scattering_forecasts = np.mean(all_lstm_scattering_sequences, axis=0)
    pure_lstm_forecasts = np.mean(all_pure_lstm_sequences, axis=0)
    sarima_forecasts = np.mean(all_sarima_sequences, axis=0)
    actual_values = np.mean(all_actual_sequences, axis=0)

    # Plotting the forecasting results for a sample of 4 randomly selected starting points within the test set
    # (of the twenty that were selcted for analysis)
    # Showing 1 week of past data (168 hours) and the full forecast period from there onwards

    sample_indices = np.random.choice(range(len(random_start_indices)), size=min(4, len(random_start_indices)), replace=False)
    sample_starts = [random_start_indices[i] for i in sample_indices]

    plt.figure(figsize=(24, 18))

    for idx, start_point in enumerate(sample_starts):
        plt.subplot(2, 2, idx + 1)
        
        context_window = 168
        start_idx = max(0, start_point - context_window)
        
        forecast_end = start_point + FORECAST_STEPS
        
        if start_idx < start_point:
            context_dates = test_data.index[start_idx:start_point]
            context_actual = test_data.iloc[start_idx:start_point].values.flatten()
            plt.plot(context_dates, context_actual, 'b-', linewidth=2, label='Historical Context', alpha=0.7)
        
        sample_idx_in_arrays = np.where(np.array(random_start_indices) == start_point)[0][0]
        
        actual_sequence = all_actual_sequences[sample_idx_in_arrays]
        lstm_scattering_sequence = all_lstm_scattering_sequences[sample_idx_in_arrays]
        pure_lstm_sequence = all_pure_lstm_sequences[sample_idx_in_arrays]
        sarima_sequence = all_sarima_sequences[sample_idx_in_arrays]
        
        forecast_dates = test_data.index[start_point:forecast_end]
        min_length = min(len(forecast_dates), len(actual_sequence), 
                        len(lstm_scattering_sequence),
                        len(pure_lstm_sequence), len(sarima_sequence))
        
        forecast_dates = forecast_dates[:min_length]
        actual_sequence = actual_sequence[:min_length]
        lstm_scattering_sequence = lstm_scattering_sequence[:min_length]
        pure_lstm_sequence = pure_lstm_sequence[:min_length]
        sarima_sequence = sarima_sequence[:min_length]
        
        plt.plot(forecast_dates, actual_sequence, 'b-', linewidth=3, label='Actual Energy', alpha=0.9)
        plt.plot(forecast_dates, lstm_scattering_sequence, 'r--', linewidth=2, label='LSTM+Wavelet', alpha=0.8)
        plt.plot(forecast_dates, pure_lstm_sequence, 'm:', linewidth=2, label='Pure LSTM', alpha=0.8)
        plt.plot(forecast_dates, sarima_sequence, 'g-', linewidth=2, label='SARIMA', alpha=0.8)
        
        if len(forecast_dates) > 0:
            plt.axvline(x=forecast_dates[0], color='red', linestyle='--', alpha=0.6, linewidth=2, label='Forecast Start')
        
        mae_lstm_scattering = np.mean(np.abs(actual_sequence - lstm_scattering_sequence))
        mae_pure_lstm = np.mean(np.abs(actual_sequence - pure_lstm_sequence))
        mae_sarima = np.mean(np.abs(actual_sequence - sarima_sequence))
        
        metrics_text = f"""Full Sequence Performance (MAE):
    LSTM+Wavelet: {mae_lstm_scattering:.1f}
    Pure LSTM: {mae_pure_lstm:.1f}
    SARIMA: {mae_sarima:.1f}

    Forecast Steps: {min_length}"""
        
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.title(f'Sample Forecast computed at: {forecast_dates[0].strftime("%Y-%m-%d %H:%M")}', 
                fontsize=12, pad=20)
        plt.xlabel('Date/Time', fontsize=11)
        plt.ylabel('Energy Load (MW)', fontsize=11)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        if len(forecast_dates) > 100:
            n_ticks = 8
            tick_indices = np.linspace(0, len(forecast_dates)-1, n_ticks, dtype=int)
            plt.xticks(forecast_dates[tick_indices], rotation=45)
        
        # Addding some padding to the y-axis here to fit the text box with the corresponding metrics
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)

    plt.tight_layout()
    plt.savefig('complete_forecast_sequences_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Looking at whether each method is best over particular hoirzons or not
    # i.e. one day forecasting, one week forecasting, one month forecasting etc etc

    print(f"\n~ Forecast metrics over different Horizons ~")
    for idx, start_point in enumerate(sample_starts):
        sample_idx_in_arrays = np.where(np.array(random_start_indices) == start_point)[0][0]
        
        actual_seq = all_actual_sequences[sample_idx_in_arrays]
        lstm_scattering_seq = all_lstm_scattering_sequences[sample_idx_in_arrays]
        pure_lstm_seq = all_pure_lstm_sequences[sample_idx_in_arrays]
        sarima_seq = all_sarima_sequences[sample_idx_in_arrays]
        
        horizons = [24, 72, 168, 336, len(actual_seq)]
        
        print(f"\nSample {idx+1} (Starting Point {start_point}):")
        print("Horizon\tLSTM+Wavelet\tPure LSTM\tSARIMA")
        
        for h in horizons:
            if h <= len(actual_seq):
                actual_h = actual_seq[:h]
                lstm_scattering_mae = np.mean(np.abs(actual_h - lstm_scattering_seq[:h]))
                pure_lstm_mae = np.mean(np.abs(actual_h - pure_lstm_seq[:h]))
                sarima_mae = np.mean(np.abs(actual_h - sarima_seq[:h]))
                
                horizon_name = f"{h}h" if h < len(actual_seq) else "Full"
                print(f"{horizon_name:>7}\t{lstm_scattering_mae:>11.1f}\t{pure_lstm_mae:>9.1f}\t{sarima_mae:>6.1f}")

    
    # Creating a plot to display degradation of these forecasting performance metrics
    
    plt.figure(figsize=(15, 10))

    horizons = [24, 48, 72, 168, 336, FORECAST_STEPS]
    avg_errors = {'LSTM+Wavelet': [], 'Pure LSTM': [], 'SARIMA': []}

    for h in horizons:
        lstm_scattering_errors = []
        pure_lstm_errors = []
        sarima_errors = []
        
        for i in range(len(all_actual_sequences)):
            if h <= len(all_actual_sequences[i]):
                actual_h = all_actual_sequences[i][:h]
                lstm_scattering_errors.append(np.mean(np.abs(actual_h - all_lstm_scattering_sequences[i][:h])))
                pure_lstm_errors.append(np.mean(np.abs(actual_h - all_pure_lstm_sequences[i][:h])))
                sarima_errors.append(np.mean(np.abs(actual_h - all_sarima_sequences[i][:h])))
        
        avg_errors['LSTM+Wavelet'].append(np.mean(lstm_scattering_errors))
        avg_errors['Pure LSTM'].append(np.mean(pure_lstm_errors))
        avg_errors['SARIMA'].append(np.mean(sarima_errors))

    plt.subplot(1, 2, 1)
    plt.plot(horizons, avg_errors['LSTM+Wavelet'], 'ro-', linewidth=2, markersize=6, label='LSTM+Wavelet')
    plt.plot(horizons, avg_errors['Pure LSTM'], 'mo-', linewidth=2, markersize=6, label='Pure LSTM')
    plt.plot(horizons, avg_errors['SARIMA'], 'go-', linewidth=2, markersize=6, label='SARIMA')

    plt.title('Forecast Error vs Prediction Horizon', fontsize=14)
    plt.xlabel('Forecast horizon in hours', fontsize=12)
    plt.ylabel('Average Mean Absolute Error in MW', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Now plotting the relative error degradation
    # i.e. how each method performs relative to the first horizon (24h)
    # The logic being that if we can optimise the first horizon for a method that has a 
    # tendency to degrade less thereafter, it could display potential

    plt.subplot(1, 2, 2)
    for model in avg_errors:
        baseline = avg_errors[model][0]
        relative_errors = [err / baseline for err in avg_errors[model]]
        
        if model == 'LSTM+Wavelet':
            plt.plot(horizons, relative_errors, 'ro-', linewidth=2, markersize=6, label=model)
        elif model == 'Pure LSTM':
            plt.plot(horizons, relative_errors, 'mo-', linewidth=2, markersize=6, label=model)
        else:
            plt.plot(horizons, relative_errors, 'go-', linewidth=2, markersize=6, label=model)

    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (24h)')
    plt.title('Relative Error Degradation', fontsize=14)
    plt.xlabel('Forecast Horizon (hours)', fontsize=12)
    plt.ylabel('Average Relative Error to the MAE at the 24 hour horizon', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_degradation_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n~~~~~~~~~~~ Summary: Average MAE by Horizon ~~~~~~~~~~~~~~~~~")
    print("Horizon\tLSTM+Wavelet\tPure LSTM\tSARIMA")
    for i, h in enumerate(horizons):
        horizon_name = f"{h}h" if h < FORECAST_STEPS else "Full"
        print(f"{horizon_name:>7}\t{avg_errors['LSTM+Wavelet'][i]:>11.1f}\t{avg_errors['Pure LSTM'][i]:>9.1f}\t{avg_errors['SARIMA'][i]:>6.1f}")

    plt.tight_layout()
    plt.savefig('sample_forecast_comparisons.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Computing a similar statistic for MAPE for greater comparability with the literature

    avg_mape = {'LSTM+Wavelet': [], 'Pure LSTM': [], 'SARIMA': []}

    for h in horizons:
        lstm_scattering_mape = []
        pure_lstm_mape = []
        sarima_mape = []
        
        for i in range(len(all_actual_sequences)):
            if h <= len(all_actual_sequences[i]):
                actual_h = all_actual_sequences[i][:h]
                
                lstm_pred_h = all_lstm_scattering_sequences[i][:h]
                non_zero_mask = actual_h != 0
                if np.any(non_zero_mask):
                    lstm_mape = np.mean(np.abs((actual_h[non_zero_mask] - lstm_pred_h[non_zero_mask]) / actual_h[non_zero_mask])) * 100
                    lstm_scattering_mape.append(lstm_mape)
                
                pure_lstm_pred_h = all_pure_lstm_sequences[i][:h]
                if np.any(non_zero_mask):
                    pure_lstm_mape_val = np.mean(np.abs((actual_h[non_zero_mask] - pure_lstm_pred_h[non_zero_mask]) / actual_h[non_zero_mask])) * 100
                    pure_lstm_mape.append(pure_lstm_mape_val)
                
                sarima_pred_h = all_sarima_sequences[i][:h]
                if np.any(non_zero_mask):
                    sarima_mape_val = np.mean(np.abs((actual_h[non_zero_mask] - sarima_pred_h[non_zero_mask]) / actual_h[non_zero_mask])) * 100
                    sarima_mape.append(sarima_mape_val)
        
        avg_mape['LSTM+Wavelet'].append(np.mean(lstm_scattering_mape) if lstm_scattering_mape else 0)
        avg_mape['Pure LSTM'].append(np.mean(pure_lstm_mape) if pure_lstm_mape else 0)
        avg_mape['SARIMA'].append(np.mean(sarima_mape) if sarima_mape else 0)

    print(f"\n~~~~~~~~~~~ Summary: Average MAPE by Horizon ~~~~~~~~~~~~~~~~~")
    print("Horizon\tLSTM+Wavelet\tPure LSTM\tSARIMA")
    for i, h in enumerate(horizons):
        horizon_name = f"{h}h" if h < FORECAST_STEPS else "Full"
        print(f"{horizon_name:>7}\t{avg_mape['LSTM+Wavelet'][i]:>11.1f}%\t{avg_mape['Pure LSTM'][i]:>8.1f}%\t{avg_mape['SARIMA'][i]:>5.1f}%")
    plt.tight_layout()
    plt.savefig('sample_forecast_comparisons.png', dpi=300, bbox_inches='tight')
    plt.show()

    # These are plots of the overally performance of the models across all starting plots collapsed into a single plot

    plt.figure(figsize=(18, 12))

    # This is a forecast plot that plots 20 points for each forecast step,
    # i.e. showing any trends in model performance across forecasts, 
    # any convergence properties or anything like that

    plt.subplot(2, 3, 1)
    sample_indices_plot = range(len(actual_values))
    plt.scatter(sample_indices_plot, actual_values, label='Actual Energy', color='blue', alpha=0.7, s=40)
    plt.scatter(sample_indices_plot, lstm_scattering_forecasts, label='LSTM+Wavelet', color='red', alpha=0.7, s=40)
    plt.scatter(sample_indices_plot, pure_lstm_forecasts, label='Pure LSTM', color='purple', alpha=0.7, s=40)
    plt.scatter(sample_indices_plot, sarima_forecasts, label='SARIMA', color='green', alpha=0.7, s=40)
    plt.title(f'Forecast Results Summary\n({NUM_RANDOM_STARTS} Random Starting Points)', fontsize=14)
    plt.xlabel('Forecast Step', fontsize=12)
    plt.ylabel('Energy (MW)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # This plot is a predictions vs actual values scatter across all test forecasts, 
    # completely abstracting away from the time dimension

    plt.subplot(2, 3, 2)
    plt.scatter(actual_values, lstm_scattering_forecasts, alpha=0.7, label='LSTM+Wavelet', color='red', s=50)
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

    # The three below plots are simple bar graph comparisons of the average forecast metrics
    # In the following order: MSE, MAE, Directional Accuracy

    plt.subplot(2, 3, 3)
    model_names = ['LSTM+Wavelet', 'Pure LSTM', 'SARIMA']
    mse_values = [avg_metrics["lstm_scattering"]["mse"], 
                    avg_metrics["pure_lstm"]["mse"], avg_metrics["sarima"]["mse"]]
    mse_stds = [avg_metrics["lstm_scattering"]["mse_std"], 
                avg_metrics["pure_lstm"]["mse_std"], avg_metrics["sarima"]["mse_std"]]

    bars = plt.bar(model_names, mse_values, yerr=mse_stds, capsize=5, color=['red', 'purple', 'green'])
    plt.title('Average Mean Squared Error Comparison', fontsize=14)
    plt.ylabel('MSE', fontsize=12)
    plt.xticks(rotation=45)

    for bar, value, std in zip(bars, mse_values, mse_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max(mse_values)*0.02, 
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 3, 4)
    mae_values = [avg_metrics["lstm_scattering"]["mae"], 
                    avg_metrics["pure_lstm"]["mae"], avg_metrics["sarima"]["mae"]]
    mae_stds = [avg_metrics["lstm_scattering"]["mae_std"], 
                avg_metrics["pure_lstm"]["mae_std"], avg_metrics["sarima"]["mae_std"]]
    bars = plt.bar(model_names, mae_values, yerr=mae_stds, capsize=5, color=['red', 'purple', 'green'])
    plt.title('Average Mean Absolute Error Comparison', fontsize=14)
    plt.ylabel('MAE', fontsize=12)
    plt.xticks(rotation=45)

    for bar, value, std in zip(bars, mae_values, mae_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max(mae_values)*0.02, 
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 3, 5)
    dir_acc_values = [avg_metrics["lstm_scattering"]["dir_acc"], 
                        avg_metrics["pure_lstm"]["dir_acc"], avg_metrics["sarima"]["dir_acc"]]
    dir_acc_stds = [avg_metrics["lstm_scattering"]["dir_acc_std"], 
                    avg_metrics["pure_lstm"]["dir_acc_std"], avg_metrics["sarima"]["dir_acc_std"]]
    bars = plt.bar(model_names, dir_acc_values, yerr=dir_acc_stds, capsize=5, color=['red', 'purple', 'green'])
    plt.title('Average Directional Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)

    for bar, value, std in zip(bars, dir_acc_values, dir_acc_stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + max(dir_acc_values)*0.02, 
                f'{value:.1f}±{std:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comprehensive_forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Lastly, the final plot looks at training times to take into account
    # The computational efficiency of each method

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    avg_training_times = [
        np.mean(training_times['lstm_scattering']),
        np.mean(training_times['pure_lstm']),
        np.mean(training_times['sarima'])
    ]
    bars = plt.bar(['LSTM+Wavelet', 'Pure LSTM', 'SARIMA'], 
            avg_training_times, color=['red', 'purple', 'green'])
    plt.title('Average Training Times', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, avg_training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_training_times)*0.02, 
                f'{value:.1f}s', ha='center', va='bottom', fontsize=11)

    plt.subplot(1, 2, 2)
    model_names = ['LSTM+Wavelet', 'Pure LSTM', 'SARIMA']
    mse_values = [avg_metrics["lstm_scattering"]["mse"], 
                    avg_metrics["pure_lstm"]["mse"], avg_metrics["sarima"]["mse"]]
    colors = ['red', 'purple', 'green']

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
        'LSTM+Wavelet': avg_metrics["lstm_scattering"]["mse"],
        'Pure LSTM': avg_metrics["pure_lstm"]["mse"],
        'SARIMA': avg_metrics["sarima"]["mse"]
    }

    best_model = min(model_mse, key=model_mse.get)
    print(f"Best performing model: {best_model} (MSE: {model_mse[best_model]:.2f})")
    
    print(f"\nCV Results Summary:")
    print(f"Best hyperparameters found: {best_config['params']}")
    print(f"Cross-validation scores: {best_config['cv_scores']}")
    print(f"Cross-validation standard deviations: {best_config['cv_std']}")