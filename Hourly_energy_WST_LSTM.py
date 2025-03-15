import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from kymatio.torch import Scattering1D
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random

class TimeSeriesDataset(Dataset):
    """
    Universal dataset class for time series forecasting that can handle:
    1. Combined scattering + raw data (for dual-input LSTM)
    2. Raw data only (for pure LSTM)
    3. Scattering coefficients only (for scattering-only models)
    
    Args:
        data (np.ndarray): Time series data as a numpy array
        window_size (int): Size of the input window
        forecast_horizon (int): Number of steps to forecast
        mode (str): One of 'combined', 'raw_only', or 'scattering_only'
        scattering_transform (kymatio.torch.Scattering1D, optional): Scattering transform object
        step (int): Step size for windowing the data, also known as the stride

    """

    def __init__(self, data, window_size, forecast_horizon, mode='combined', 
                 scattering_transform=None, step=1):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        self.scattering = scattering_transform
        self.step = step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        valid_modes = ['combined', 'raw_only', 'scattering_only']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
        
        if (mode == 'combined' or mode == 'scattering_only') and scattering_transform is None:
            raise ValueError(f"Scattering transform must be provided for mode '{mode}'")
        
        self.indices = list(range(0, len(data) - window_size - forecast_horizon + 1, step))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        window = self.data[start_idx:start_idx + self.window_size].flatten()
        target = self.data[start_idx + self.window_size:start_idx + self.window_size + self.forecast_horizon].flatten()
        
        window_tensor = torch.tensor(window, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        if self.mode == 'raw_only':
            raw_window = self.data[start_idx:start_idx + self.window_size]
            raw_tensor = torch.tensor(raw_window, dtype=torch.float32)
            return raw_tensor.reshape(-1, 1), target_tensor
        
        with torch.no_grad():
            scattering_input = window_tensor.unsqueeze(0).unsqueeze(0)
            scattering_coeffs = self.scattering(scattering_input)
            
            if self.mode == 'combined':
                scattering_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[0], 
                                                             scattering_coeffs.shape[2], 1)
                return scattering_coeffs[0], window_tensor.reshape(-1, 1), target_tensor
            
            elif self.mode == 'scattering_only':
                scattering_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[2], 1)
                return scattering_coeffs, target_tensor


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

def rolling_window_forecast_scattering_lstm(train_data, test_data, window_size, forecast_horizon, scattering_params, lstm_params, random_seed=42):
    """
    Optimized version: Generates rolling window forecasts using Wavelet Scattering 
    Transform and LSTM, Pure LSTM, Scattering-Only LSTM, and ARIMA for large datasets
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    
    print(f"Random seed set to: {random_seed}")
    
    lstm_scattering_forecasts = []
    pure_lstm_forecasts = []
    scattering_only_forecasts = []
    arima_forecasts = []
    actual_values = []
    lstm_scattering_training_times = []
    pure_lstm_training_times = []
    scattering_only_training_times = []
    arima_training_times = []

    # Fitting the scaler only on the training data, for now
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    
    # Dataset here is 27k data points, so just to speed up training have limited it to 8k
    if len(train_data) > 10000:
        print(f"Training on last 8000 points of {len(train_data)} total points")
        train_data_scaled = train_data_scaled[-8000:]

    # Initializing the wavelet scattering transform, outside of the training loop
    J = min(int(np.log2(window_size)), 8)
    print(f"Using J={J} for scattering transform")
    
    scattering_params['J'] = J
    scattering = Scattering1D(**scattering_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    scattering.to(device)
    
    # Defining the LSTM model - using a dual input architecture
    # Basically, computing a separate LSTM for the scattering coefficients and raw time series
    # And combining them in a single fully connected layer
    class ScatteringLSTM(nn.Module):
        def __init__(self, scattering_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
            super(ScatteringLSTM, self).__init__()
            self.lstm_scattering = nn.LSTM(1, hidden_dim, num_layers=num_layers, 
                                         batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
            self.lstm_raw = nn.LSTM(1, hidden_dim, num_layers=num_layers, 
                                  batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, scattering_x, raw_x):
            scattering_out, _ = self.lstm_scattering(scattering_x)
            scattering_out = scattering_out[:, -1, :]

            raw_out, _ = self.lstm_raw(raw_x)
            raw_out = raw_out[:, -1, :]

            combined = torch.cat((scattering_out, raw_out), dim=1)
            combined = self.dropout(combined)
            output = self.fc(combined)
            return output
    
    # Defining the class for the Second LSTM model, that only takes the raw time series as input
    class PureLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
            super(PureLSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            lstm_out = self.dropout(lstm_out)
            output = self.fc(lstm_out)
            return output
    
    # Defining the class for the Third LSTM model, which only takes the scattering coefficients as inputs
    class ScatteringOnlyLSTM(nn.Module):
        def __init__(self, scattering_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
            super(ScatteringOnlyLSTM, self).__init__()
            self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            lstm_out = self.dropout(lstm_out)
            output = self.fc(lstm_out)
            return output
    
    step_size = 4
    train_dataset = TimeSeriesDataset(
        train_data_scaled, 
        window_size, 
        forecast_horizon,
        mode='combined',
        scattering_transform=scattering.to(device),
        step=step_size
    )
    print(f"Training dataset contains {len(train_dataset)} samples with step size {step_size}")
    
    # Creating alernative dataset for pure LSTM (without scattering coefficient adjustment)
    pure_lstm_dataset = TimeSeriesDataset(
        train_data_scaled, 
        window_size, 
        forecast_horizon,
        mode='raw_only',
        scattering_transform=None, 
        step=step_size
    )
    
    # Creating an alternative dataset for the scattering-only LSTM model 
    # (no raw data points, only scattering coefficients)
    scattering_only_dataset = TimeSeriesDataset(
        train_data_scaled,
        window_size,
        forecast_horizon,
        mode='scattering_only',
        scattering_transform=scattering.to(device),
        step=step_size
    )
    
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    pure_lstm_loader = DataLoader(
        pure_lstm_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    scattering_only_loader = DataLoader(
        scattering_only_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    sample_batch = next(iter(train_loader))
    sample_scattering, sample_raw, _ = sample_batch
    
    scattering_shape = sample_scattering.shape[1]
    print(f"Scattering shape: {scattering_shape}")
    
    scattering_only_sample = next(iter(scattering_only_loader))
    scattering_only_input, _ = scattering_only_sample
    
    # Initialize the models
    lstm_scattering_model = ScatteringLSTM(
        scattering_dim=scattering_shape,
        hidden_dim=lstm_params['hidden_dim'],
        output_dim=forecast_horizon,
        dropout_rate=lstm_params['dropout_rate'],
        num_layers=lstm_params['num_layers']
    )
    lstm_scattering_model.to(device)
    
    pure_lstm_model = PureLSTM(
        input_dim=1,
        hidden_dim=lstm_params['hidden_dim'],
        output_dim=forecast_horizon,
        dropout_rate=lstm_params['dropout_rate'],
        num_layers=lstm_params['num_layers']
    )
    pure_lstm_model.to(device)
    
    scattering_only_model = ScatteringOnlyLSTM(
        scattering_dim=scattering_only_input.shape[1],
        hidden_dim=lstm_params['hidden_dim'],
        output_dim=forecast_horizon,
        dropout_rate=lstm_params['dropout_rate'],
        num_layers=lstm_params['num_layers']
    )
    scattering_only_model.to(device)
    
    print(f"LSTM+Scattering Model Architecture:")
    print(f"  Hidden Dimensions: {lstm_params['hidden_dim']}")
    print(f"  Number of Layers: {lstm_params['num_layers']}")
    print(f"  Dropout Rate: {lstm_params['dropout_rate']}")
    print(f"  Total Parameters: {sum(p.numel() for p in lstm_scattering_model.parameters())}")
    
    print(f"\nPure LSTM Model Architecture:")
    print(f"  Hidden Dimensions: {lstm_params['hidden_dim']}")
    print(f"  Number of Layers: {lstm_params['num_layers']}")
    print(f"  Dropout Rate: {lstm_params['dropout_rate']}")
    print(f"  Total Parameters: {sum(p.numel() for p in pure_lstm_model.parameters())}")
    
    print(f"\nScattering-Only LSTM Model Architecture:")
    print(f"  Hidden Dimensions: {lstm_params['hidden_dim']}")
    print(f"  Number of Layers: {lstm_params['num_layers']}")
    print(f"  Dropout Rate: {lstm_params['dropout_rate']}")
    print(f"  Total Parameters: {sum(p.numel() for p in scattering_only_model.parameters())}")
    
    optimizer_scattering = optim.Adam(lstm_scattering_model.parameters(), lr=lstm_params['learning_rate'])
    optimizer_pure = optim.Adam(pure_lstm_model.parameters(), lr=lstm_params['learning_rate'])
    optimizer_scattering_only = optim.Adam(scattering_only_model.parameters(), lr=lstm_params['learning_rate'])  # New optimizer
    criterion = nn.MSELoss()
    
    lstm_scattering_start_time = time.time()
    lstm_scattering_model.train()
    
    early_stop_patience = 5
    best_loss_scattering = float('inf')
    patience_counter_scattering = 0
    
    print("\n===== Training LSTM+Scattering Model =====")
    for epoch in range(lstm_params['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        for scattering_batch, raw_batch, targets_batch in train_loader:
            batch_count += 1
            
            scattering_batch = scattering_batch.to(device)
            raw_batch = raw_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer_scattering.zero_grad()
            
            outputs = lstm_scattering_model(scattering_batch, raw_batch)
            loss = criterion(outputs, targets_batch)
            
            loss.backward()
            # Gradient clipping is useed here to to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(lstm_scattering_model.parameters(), 1.0)
            optimizer_scattering.step()
            
            epoch_loss += loss.item()
            
            # Printing batch progress at intervals of 20, since it takes forever to train and I get scared it is crashing :))
            if batch_count % 20 == 0:
                print(f"Epoch {epoch+1}/{lstm_params['epochs']}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{lstm_params['epochs']} complete. Avg loss: {avg_epoch_loss:.6f}")
        
        if avg_epoch_loss < best_loss_scattering:
            best_loss_scattering = avg_epoch_loss
            patience_counter_scattering = 0
            torch.save(lstm_scattering_model.state_dict(), 'best_energy_forecast_scattering_model.pt')
        else:
            patience_counter_scattering += 1
            print(f"Early stopping patience: {patience_counter_scattering}/{early_stop_patience}")
            if patience_counter_scattering >= early_stop_patience:
                print("Early stopping triggered!")
                break
    
    # Loading the best model
    lstm_scattering_model.load_state_dict(torch.load('best_energy_forecast_scattering_model.pt'))
    
    lstm_scattering_training_time = time.time() - lstm_scattering_start_time
    lstm_scattering_training_times.append(lstm_scattering_training_time)
    print(f"LSTM+Scattering training completed in {lstm_scattering_training_time:.2f} seconds")
    
    # Training the Pure LSTM model
    pure_lstm_start_time = time.time()
    pure_lstm_model.train()
    
    best_loss_pure = float('inf')
    patience_counter_pure = 0
    
    print("\n===== Training Pure LSTM Model =====")
    for epoch in range(lstm_params['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        for inputs_batch, targets_batch in pure_lstm_loader:
            batch_count += 1
            
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer_pure.zero_grad()
            
            outputs = pure_lstm_model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pure_lstm_model.parameters(), 1.0)
            optimizer_pure.step()
            
            epoch_loss += loss.item()
            
            if batch_count % 20 == 0:
                print(f"Epoch {epoch+1}/{lstm_params['epochs']}, Batch {batch_count}/{len(pure_lstm_loader)}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{lstm_params['epochs']} complete. Avg loss: {avg_epoch_loss:.6f}")
        
        if avg_epoch_loss < best_loss_pure:
            best_loss_pure = avg_epoch_loss
            patience_counter_pure = 0
            torch.save(pure_lstm_model.state_dict(), 'best_energy_forecast_pure_lstm_model.pt')
        else:
            patience_counter_pure += 1
            print(f"Early stopping patience: {patience_counter_pure}/{early_stop_patience}")
            if patience_counter_pure >= early_stop_patience:
                print("Early stopping triggered!")
                break
    
    # Loading the best PURE LSTM model
    pure_lstm_model.load_state_dict(torch.load('best_energy_forecast_pure_lstm_model.pt'))
    
    pure_lstm_training_time = time.time() - pure_lstm_start_time
    pure_lstm_training_times.append(pure_lstm_training_time)
    print(f"Pure LSTM training completed in {pure_lstm_training_time:.2f} seconds")
    
    # Finally, Training the Scattering-Only LSTM model
    scattering_only_start_time = time.time()
    scattering_only_model.train()
    
    best_loss_scattering_only = float('inf')
    patience_counter_scattering_only = 0
    
    print("\n===== Training Scattering-Only LSTM Model =====")
    for epoch in range(lstm_params['epochs']):
        epoch_loss = 0
        batch_count = 0
        
        for inputs_batch, targets_batch in scattering_only_loader:
            batch_count += 1
            
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer_scattering_only.zero_grad()
            
            outputs = scattering_only_model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scattering_only_model.parameters(), 1.0)
            optimizer_scattering_only.step()
            
            epoch_loss += loss.item()
            
            if batch_count % 20 == 0:
                print(f"Epoch {epoch+1}/{lstm_params['epochs']}, Batch {batch_count}/{len(scattering_only_loader)}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{lstm_params['epochs']} complete. Avg loss: {avg_epoch_loss:.6f}")
        
        if avg_epoch_loss < best_loss_scattering_only:
            best_loss_scattering_only = avg_epoch_loss
            patience_counter_scattering_only = 0
            torch.save(scattering_only_model.state_dict(), 'best_energy_forecast_scattering_only_model.pt')
        else:
            patience_counter_scattering_only += 1
            print(f"Early stopping patience: {patience_counter_scattering_only}/{early_stop_patience}")
            if patience_counter_scattering_only >= early_stop_patience:
                print("Early stopping triggered!")
                break
    
    # Loading the best pure scattering model
    scattering_only_model.load_state_dict(torch.load('best_energy_forecast_scattering_only_model.pt'))
    
    scattering_only_training_time = time.time() - scattering_only_start_time
    scattering_only_training_times.append(scattering_only_training_time)
    print(f"Scattering-Only LSTM training completed in {scattering_only_training_time:.2f} seconds")
    
    # Finally, returning to the test data and scaling it
    test_data_scaled = scaler.transform(test_data)
    
    # For ARIMA, currenlty using a pre-selected model to save time
    # Planning to adjus the code to select a SARIMA mode based on the data using lowest AIC/BIC score
    best_order = (2, 1, 1)
    print(f"Using ARIMA{best_order} for forecasting")
    
    # Setting a smaller window for the ARIMA, for obvious reasons
    arima_window_size = min(window_size, 48)
    
    print("Starting test forecasting...")
    total_forecast_points = len(test_data)
    
    full_data_scaled = np.concatenate([train_data_scaled, test_data_scaled])
    orig_train_data = train_data.values.flatten()
    
    for i in range(total_forecast_points):
        train_end_idx = len(train_data_scaled) + i - 1
        window_start_idx = train_end_idx - window_size + 1
        current_window = full_data_scaled[window_start_idx:train_end_idx+1].flatten()
        
        actual = test_data.iloc[i].values.flatten()
        actual_values.append(actual[0])
        
        window_tensor = torch.tensor(current_window, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Reshaping for scattering transform
            scattering_input = window_tensor.reshape(1, 1, -1)
            scattering_coeffs = scattering(scattering_input)
            
            # Reshaping again, for the LSTM and scattering input
            scattering_coef_input = scattering_coeffs.reshape(1, scattering_coeffs.shape[2], 1)
            raw_input = window_tensor.reshape(1, -1, 1)
            
            # Doing the forecast with the dual input
            lstm_scattering_model.eval()
            lstm_scattering_pred_scaled = lstm_scattering_model(scattering_coef_input, raw_input).cpu().numpy().flatten()
            
            lstm_scattering_pred = scaler.inverse_transform(lstm_scattering_pred_scaled.reshape(-1, 1)).flatten()
            lstm_scattering_forecasts.append(lstm_scattering_pred[0])
            
            # Doing the pure LSTM forecast
            pure_lstm_model.eval()
            pure_lstm_input = window_tensor.reshape(1, -1, 1)
            pure_lstm_pred_scaled = pure_lstm_model(pure_lstm_input).cpu().numpy().flatten()
            
            pure_lstm_pred = scaler.inverse_transform(pure_lstm_pred_scaled.reshape(-1, 1)).flatten()
            pure_lstm_forecasts.append(pure_lstm_pred[0])
            
            # Doing the pure scattering LSTM forecast
            scattering_only_model.eval()
            scattering_only_input = scattering_coeffs.reshape(1, scattering_coeffs.shape[2], 1)
            scattering_only_pred_scaled = scattering_only_model(scattering_only_input).cpu().numpy().flatten()
            
            scattering_only_pred = scaler.inverse_transform(scattering_only_pred_scaled.reshape(-1, 1)).flatten()
            scattering_only_forecasts.append(scattering_only_pred[0])
        
        arima_start_time = time.time()
        
        orig_values = np.concatenate([
            orig_train_data,
            test_data.iloc[:i].values.flatten() if i > 0 else []
        ])
        orig_window = orig_values[-arima_window_size:]
        
        try:
            arima_model = ARIMA(pd.Series(orig_window), order=best_order)
            arima_result = arima_model.fit()
            arima_pred = arima_result.forecast(steps=forecast_horizon)
            arima_forecasts.append(arima_pred[0])
        except Exception as e:
            print(f"ARIMA error at idx {i}: {str(e)}")
            arima_forecasts.append(orig_window[-1])
        
        arima_training_time = time.time() - arima_start_time
        arima_training_times.append(arima_training_time)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Completed forecast {i+1}/{total_forecast_points}")
            print(f"LSTM+Scattering: {lstm_scattering_forecasts[-1]:.2f}, Scattering-Only: {scattering_only_forecasts[-1]:.2f}, Pure LSTM: {pure_lstm_forecasts[-1]:.2f}, Actual: {actual[0]:.2f}")
    
    weight_analysis = analyze_model_weights(lstm_scattering_model, lstm_params)
    
    training_times = {
        'lstm_scattering': lstm_scattering_training_times, 
        'scattering_only': scattering_only_training_times,
        'pure_lstm': pure_lstm_training_times, 
        'arima': arima_training_times
    }
    return lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, actual_values, arima_forecasts, training_times, lstm_scattering_model, pure_lstm_model, scattering_only_model

if __name__ == "__main__":
    global_seed = 42 # Claaaaaaasic
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    df = pd.read_excel("/Users/jakubfridrich/Documents/Hourly_energy_c_2022-25.xlsx")
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
    print(f"Global random seed: {global_seed}")
    start_time = time.time()
    lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, actual_values, arima_forecasts, training_times, lstm_scattering_model, pure_lstm_model, scattering_only_model = rolling_window_forecast_scattering_lstm(
        train_data, test_data, window_size, forecast_horizon, scattering_params, lstm_params, random_seed=global_seed
    )
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    metrics = evaluate_forecasts(actual_values, lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, arima_forecasts)
    
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