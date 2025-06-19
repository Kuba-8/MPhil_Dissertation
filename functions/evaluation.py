import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecasts(actual_values, lstm_scattering_forecasts, pure_lstm_forecasts, sarima_forecasts):
    """
    Just a load of forecast evaluation metrics
    """

    lstm_scattering_mse = mean_squared_error(actual_values, lstm_scattering_forecasts)
    lstm_scattering_mae = mean_absolute_error(actual_values, lstm_scattering_forecasts)
    lstm_scattering_rmse = np.sqrt(lstm_scattering_mse)
    
    pure_lstm_mse = mean_squared_error(actual_values, pure_lstm_forecasts)
    pure_lstm_mae = mean_absolute_error(actual_values, pure_lstm_forecasts)
    pure_lstm_rmse = np.sqrt(pure_lstm_mse)
    
    sarima_mse = mean_squared_error(actual_values, sarima_forecasts)
    sarima_mae = mean_absolute_error(actual_values, sarima_forecasts)
    sarima_rmse = np.sqrt(sarima_mse)
    
    def mape(actual, pred):
        return np.mean(np.abs((np.array(actual) - np.array(pred)) / np.array(actual))) * 100
    
    lstm_scattering_mape = mape(actual_values, lstm_scattering_forecasts)
    pure_lstm_mape = mape(actual_values, pure_lstm_forecasts)
    sarima_mape = mape(actual_values, sarima_forecasts)
    
    # Directional accuracy = whether it is predicting up/down movement correctly
    def directional_accuracy(actual, pred):
        actual_diff = np.diff(actual)
        pred_diff = np.diff(pred)
        correct_dir = np.sum((actual_diff > 0) == (pred_diff > 0))
        return correct_dir / len(actual_diff) * 100
    
    lstm_scattering_dir_acc = directional_accuracy(actual_values, lstm_scattering_forecasts)
    pure_lstm_dir_acc = directional_accuracy(actual_values, pure_lstm_forecasts)
    sarima_dir_acc = directional_accuracy(actual_values, sarima_forecasts)
    
    print("\n~~Comprehensive Forecast Evaluation~~")
    print(f"LSTM+Scattering Metrics:")
    print(f"  MSE: {lstm_scattering_mse:.2f}")
    print(f"  RMSE: {lstm_scattering_rmse:.2f}")
    print(f"  MAE: {lstm_scattering_mae:.2f}")
    print(f"  MAPE: {lstm_scattering_mape:.2f}%")
    print(f"  Directional Accuracy: {lstm_scattering_dir_acc:.2f}%")
    
    print(f"\nPure LSTM Metrics:")
    print(f"  MSE: {pure_lstm_mse:.2f}")
    print(f"  RMSE: {pure_lstm_rmse:.2f}")
    print(f"  MAE: {pure_lstm_mae:.2f}")
    print(f"  MAPE: {pure_lstm_mape:.2f}%")
    print(f"  Directional Accuracy: {pure_lstm_dir_acc:.2f}%")
    
    print(f"\nSARIMA Metrics:")
    print(f"  MSE: {sarima_mse:.2f}")
    print(f"  RMSE: {sarima_rmse:.2f}")
    print(f"  MAE: {sarima_mae:.2f}")
    print(f"  MAPE: {sarima_mape:.2f}%")
    print(f"  Directional Accuracy: {sarima_dir_acc:.2f}%")
    
    print(f"\nRelative Improvements:")
    print(f"  LSTM+Scattering vs SARIMA:")
    print(f"    MSE: {(sarima_mse - lstm_scattering_mse) / sarima_mse * 100:.2f}% lower with LSTM+Scattering")
    print(f"    MAE: {(sarima_mae - lstm_scattering_mae) / sarima_mae * 100:.2f}% lower with LSTM+Scattering")
    
    print(f"\n  Pure LSTM vs SARIMA:")
    print(f"    MSE: {(sarima_mse - pure_lstm_mse) / sarima_mse * 100:.2f}% lower with Pure LSTM")
    print(f"    MAE: {(sarima_mae - pure_lstm_mae) / sarima_mae * 100:.2f}% lower with Pure LSTM")
    
    print(f"\n  LSTM+Scatering vs Pure LSTM:")
    print(f"    MSE: {(pure_lstm_mse - lstm_scattering_mse) / pure_lstm_mse * 100:.2f}% lower with LSTM+Scattering")
    print(f"    MAE: {(pure_lstm_mae - lstm_scattering_mae) / pure_lstm_mae * 100:.2f}% lower with LSTM+Scattering")
    
    return {
        'lstm_scattering': {
            'mse': lstm_scattering_mse,
            'rmse': lstm_scattering_rmse,
            'mae': lstm_scattering_mae,
            'mape': lstm_scattering_mape,
            'dir_acc': lstm_scattering_dir_acc
        },
        'pure_lstm': {
            'mse': pure_lstm_mse,
            'rmse': pure_lstm_rmse,
            'mae': pure_lstm_mae,
            'mape': pure_lstm_mape,
            'dir_acc': pure_lstm_dir_acc
        },
        'sarima': {
            'mse': sarima_mse,
            'rmse': sarima_rmse,
            'mae': sarima_mae,
            'mape': sarima_mape,
            'dir_acc': sarima_dir_acc
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
    fc_weights = lstm_model.fc1.weight.data.cpu().numpy()

    # First half correspond to the scattering LSTM
    # Second half correspond to the raw data LSTM
    scattering_hidden_dim = lstm_params['hidden_dim'] // 2
    scattering_weights = fc_weights[:, :scattering_hidden_dim]
    raw_data_weights = fc_weights[:, scattering_hidden_dim:scattering_hidden_dim + lstm_params['hidden_dim']]

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