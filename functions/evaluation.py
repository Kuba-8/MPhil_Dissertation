import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecasts(actual_values, lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, arima_forecasts):
    """
    Just a load of forecast evaluation metrics
    """

    lstm_scattering_mse = mean_squared_error(actual_values, lstm_scattering_forecasts)
    lstm_scattering_mae = mean_absolute_error(actual_values, lstm_scattering_forecasts)
    lstm_scattering_rmse = np.sqrt(lstm_scattering_mse)
    
    pure_lstm_mse = mean_squared_error(actual_values, pure_lstm_forecasts)
    pure_lstm_mae = mean_absolute_error(actual_values, pure_lstm_forecasts)
    pure_lstm_rmse = np.sqrt(pure_lstm_mse)
    
    scattering_only_mse = mean_squared_error(actual_values, scattering_only_forecasts)
    scattering_only_mae = mean_absolute_error(actual_values, scattering_only_forecasts)
    scattering_only_rmse = np.sqrt(scattering_only_mse)
    
    arima_mse = mean_squared_error(actual_values, arima_forecasts)
    arima_mae = mean_absolute_error(actual_values, arima_forecasts)
    arima_rmse = np.sqrt(arima_mse)
    
    def mape(actual, pred):
        return np.mean(np.abs((np.array(actual) - np.array(pred)) / np.array(actual))) * 100
    
    lstm_scattering_mape = mape(actual_values, lstm_scattering_forecasts)
    pure_lstm_mape = mape(actual_values, pure_lstm_forecasts)
    scattering_only_mape = mape(actual_values, scattering_only_forecasts)
    arima_mape = mape(actual_values, arima_forecasts)
    
    # Directional accuracy = whether it is predicting up/down movement correctly
    def directional_accuracy(actual, pred):
        actual_diff = np.diff(actual)
        pred_diff = np.diff(pred)
        correct_dir = np.sum((actual_diff > 0) == (pred_diff > 0))
        return correct_dir / len(actual_diff) * 100
    
    lstm_scattering_dir_acc = directional_accuracy(actual_values, lstm_scattering_forecasts)
    pure_lstm_dir_acc = directional_accuracy(actual_values, pure_lstm_forecasts)
    scattering_only_dir_acc = directional_accuracy(actual_values, scattering_only_forecasts)
    arima_dir_acc = directional_accuracy(actual_values, arima_forecasts)
    
    print("\n===== Comprehensive Forecast Evaluation =====")
    print(f"LSTM+Scattering Metrics:")
    print(f"  MSE: {lstm_scattering_mse:.2f}")
    print(f"  RMSE: {lstm_scattering_rmse:.2f}")
    print(f"  MAE: {lstm_scattering_mae:.2f}")
    print(f"  MAPE: {lstm_scattering_mape:.2f}%")
    print(f"  Directional Accuracy: {lstm_scattering_dir_acc:.2f}%")
    
    print(f"\nScattering-Only LSTM Metrics:")
    print(f"  MSE: {scattering_only_mse:.2f}")
    print(f"  RMSE: {scattering_only_rmse:.2f}")
    print(f"  MAE: {scattering_only_mae:.2f}")
    print(f"  MAPE: {scattering_only_mape:.2f}%")
    print(f"  Directional Accuracy: {scattering_only_dir_acc:.2f}%")
    
    print(f"\nPure LSTM Metrics:")
    print(f"  MSE: {pure_lstm_mse:.2f}")
    print(f"  RMSE: {pure_lstm_rmse:.2f}")
    print(f"  MAE: {pure_lstm_mae:.2f}")
    print(f"  MAPE: {pure_lstm_mape:.2f}%")
    print(f"  Directional Accuracy: {pure_lstm_dir_acc:.2f}%")
    
    print(f"\nARIMA Metrics:")
    print(f"  MSE: {arima_mse:.2f}")
    print(f"  RMSE: {arima_rmse:.2f}")
    print(f"  MAE: {arima_mae:.2f}")
    print(f"  MAPE: {arima_mape:.2f}%")
    print(f"  Directional Accuracy: {arima_dir_acc:.2f}%")
    
    print(f"\nRelative Improvements:")
    print(f"  LSTM+Scattering vs ARIMA:")
    print(f"    MSE: {(arima_mse - lstm_scattering_mse) / arima_mse * 100:.2f}% lower with LSTM+Scattering")
    print(f"    MAE: {(arima_mae - lstm_scattering_mae) / arima_mae * 100:.2f}% lower with LSTM+Scattering")
    
    print(f"\n  Scattering-Only LSTM vs ARIMA:")
    print(f"    MSE: {(arima_mse - scattering_only_mse) / arima_mse * 100:.2f}% lower with Scattering-Only LSTM")
    print(f"    MAE: {(arima_mae - scattering_only_mae) / arima_mae * 100:.2f}% lower with Scattering-Only LSTM")
    
    print(f"\n  Pure LSTM vs ARIMA:")
    print(f"    MSE: {(arima_mse - pure_lstm_mse) / arima_mse * 100:.2f}% lower with Pure LSTM")
    print(f"    MAE: {(arima_mae - pure_lstm_mae) / arima_mae * 100:.2f}% lower with Pure LSTM")
    
    print(f"\n  LSTM+Scattering vs Pure LSTM:")
    print(f"    MSE: {(pure_lstm_mse - lstm_scattering_mse) / pure_lstm_mse * 100:.2f}% lower with LSTM+Scattering")
    print(f"    MAE: {(pure_lstm_mae - lstm_scattering_mae) / pure_lstm_mae * 100:.2f}% lower with LSTM+Scattering")
    
    print(f"\n  LSTM+Scattering vs Scattering-Only LSTM:")
    print(f"    MSE: {(scattering_only_mse - lstm_scattering_mse) / scattering_only_mse * 100:.2f}% lower with LSTM+Scattering")
    print(f"    MAE: {(scattering_only_mae - lstm_scattering_mae) / scattering_only_mae * 100:.2f}% lower with LSTM+Scattering")
    
    print(f"\n  Scattering-Only LSTM vs Pure LSTM:")
    print(f"    MSE: {(pure_lstm_mse - scattering_only_mse) / pure_lstm_mse * 100:.2f}% lower with Scattering-Only LSTM")
    print(f"    MAE: {(pure_lstm_mae - scattering_only_mae) / pure_lstm_mae * 100:.2f}% lower with Scattering-Only LSTM")
    
    return {
        'lstm_scattering': {
            'mse': lstm_scattering_mse,
            'rmse': lstm_scattering_rmse,
            'mae': lstm_scattering_mae,
            'mape': lstm_scattering_mape,
            'dir_acc': lstm_scattering_dir_acc
        },
        'scattering_only': {
            'mse': scattering_only_mse,
            'rmse': scattering_only_rmse,
            'mae': scattering_only_mae,
            'mape': scattering_only_mape,
            'dir_acc': scattering_only_dir_acc
        },
        'pure_lstm': {
            'mse': pure_lstm_mse,
            'rmse': pure_lstm_rmse,
            'mae': pure_lstm_mae,
            'mape': pure_lstm_mape,
            'dir_acc': pure_lstm_dir_acc
        },
        'arima': {
            'mse': arima_mse,
            'rmse': arima_rmse,
            'mae': arima_mae,
            'mape': arima_mape,
            'dir_acc': arima_dir_acc
        }
    }


def analyze_model_weights(lstm_model, lstm_params):
    """
    Analyze the weights of the fully connected layer to determine the relative importance 
    of scattering coefficients vs. raw data
    """
    print("\n===== Analyzing Model Weights =====")
    lstm_model.eval()
    
    # Extracting the weights from the fully connected layer
    fc_weights = lstm_model.fc.weight.data.cpu().numpy()

    # First half correspond to the scattering LSTM
    # Second half correspond to the raw data LSTM
    scattering_weights = fc_weights[:, :lstm_params['hidden_dim']]
    raw_data_weights = fc_weights[:, lstm_params['hidden_dim']:]

    # Calcu the absolute sum of weights for each component
    # (= overall influence of each component)
    scattering_influence = np.abs(scattering_weights).sum()
    raw_data_influence = np.abs(raw_data_weights).sum()

    # Calc the relative importance
    total_influence = scattering_influence + raw_data_influence
    scattering_importance = scattering_influence / total_influence * 100
    raw_data_importance = raw_data_influence / total_influence * 100

    print(f"\n===== Fully Connected Layer Weight Analysis =====")
    print(f"Scattering LSTM total absolute weight: {scattering_influence:.4f}")
    print(f"Raw data LSTM total absolute weight: {raw_data_influence:.4f}")
    print(f"Scattering LSTM relative importance: {scattering_importance:.2f}%")
    print(f"Raw data LSTM relative importance: {raw_data_importance:.2f}%")

    scattering_abs_mean = np.abs(scattering_weights).mean()
    raw_data_abs_mean = np.abs(raw_data_weights).mean()
    scattering_abs_std = np.abs(scattering_weights).std()
    raw_data_abs_std = np.abs(raw_data_weights).std()

    print(f"\nScattering LSTM mean absolute weight: {scattering_abs_mean:.4f} (±{scattering_abs_std:.4f})")
    print(f"Raw data LSTM mean absolute weight: {raw_data_abs_mean:.4f} (±{raw_data_abs_std:.4f})")

    plt.figure(figsize=(12, 8))

    # Relative importance plot
    plt.subplot(2, 2, 1)
    importance = [scattering_importance, raw_data_importance]
    plt.bar(['Scattering LSTM', 'Raw Data LSTM'], importance, color=['#3498db', '#e74c3c'])
    plt.title('Relative Importance (%)')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)

    # Weight distribution histograms
    plt.subplot(2, 2, 2)
    plt.hist(scattering_weights.flatten(), bins=30, alpha=0.7, label='Scattering LSTM', color='#3498db')
    plt.hist(raw_data_weights.flatten(), bins=30, alpha=0.7, label='Raw Data LSTM', color='#e74c3c')
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Heatmap of scattering weights
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(scattering_weights), cmap='Blues', aspect='auto')
    plt.colorbar(label='Absolute Weight Value')
    plt.title('Scattering LSTM Weights')
    plt.xlabel('Hidden Unit')
    plt.ylabel('Output Unit')

    # Heatmap of raw data weights
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(raw_data_weights), cmap='Reds', aspect='auto')
    plt.colorbar(label='Absolute Weight Value')
    plt.title('Raw Data LSTM Weights')
    plt.xlabel('Hidden Unit')
    plt.ylabel('Output Unit')

    plt.tight_layout()
    plt.savefig('weight_analysis.png', dpi=300)
    plt.show()
    
    return {
        'scattering_importance': scattering_importance,
        'raw_data_importance': raw_data_importance,
        'scattering_weights': scattering_weights,
        'raw_data_weights': raw_data_weights
    }