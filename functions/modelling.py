import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from kymatio.torch import Scattering1D
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import warnings

from functions.data_processing import TimeSeriesDataset
from functions.evaluation import analyze_model_weights

def find_optimal_sarima(data, max_p=3, max_d=2, max_q=3, max_P=2, max_D=1, max_Q=2, seasonal_period=24):
    """
    Simple function for finding the optimal SARIMA parameters using grid search with AIC criterion

    """
    warnings.filterwarnings("ignore")
    
    # Generate each and every parameter combination
    # P, D, Q are seasonal parameters, whereas p, d, q are non-seasonal, 
    # standard ARIMA parameters

    p = d = q = range(0, max_p + 1)
    P = D = Q = range(0, max_P + 1)
    
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q, [seasonal_period]))
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    print("Searching for optimal SARIMA parameters...")
    tested_count = 0
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(data, order=param, seasonal_order=param_seasonal)
                results = model.fit(disp=False, maxiter=50)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = param_seasonal
                
                tested_count += 1
                if tested_count % 20 == 0:
                    print(f"Tested {tested_count} parameter combinations...")
                    
            except Exception:
                continue
    
    print(f"Best SARIMA{best_params}x{best_seasonal_params} with AIC: {best_aic:.2f}")
    return best_params, best_seasonal_params

# Defining the LSTM model - using a dual input architecture
# Basically, computing a separate LSTM for the scattering coefficients and raw time series
# And combining them in a single fully connected layer
# Now though, using more fully connected layers and fewer LSTM layers essentially, motivated by the literature (see paper)
class ScatteringLSTM(nn.Module):
    def __init__(self, scattering_dim, lag_features_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
        super(ScatteringLSTM, self).__init__()
        self.lstm_scattering = nn.LSTM(1, hidden_dim, num_layers=num_layers, 
                                      batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.lstm_raw = nn.LSTM(lag_features_dim, hidden_dim, num_layers=num_layers, 
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 50)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(50, output_dim)

    def forward(self, scattering_x, raw_x):
        scattering_out, _ = self.lstm_scattering(scattering_x)
        scattering_out = scattering_out[:, -1, :]

        raw_out, _ = self.lstm_raw(raw_x)
        raw_out = raw_out[:, -1, :]

        combined = torch.cat((scattering_out, raw_out), dim=1)
        combined = self.dropout(combined)
        
        x = self.relu1(self.fc1(combined))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        output = self.fc_out(x)
        
        return output

# Defining the class for the Second LSTM model, that only takes the raw time series as input
# Architecture has similarly changed to a more standard LSTM
# Based on the literature essentially, decided prior model was very poorly motivated
class PureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
        super(PureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 50)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(50, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        x = self.dropout(lstm_out)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        output = self.fc_out(x)
        
        return output

# Defining the class for the Third LSTM model, which only takes the scattering coefficients as inputs
# Architecture has similarly changed to a more standard LSTM
# Based on the literature essentially, decided prior model was very poorly motivated
class ScatteringOnlyLSTM(nn.Module):
    def __init__(self, scattering_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
        super(ScatteringOnlyLSTM, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 50)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(50, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        x = self.dropout(lstm_out)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        output = self.fc_out(x)
        
        return output

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
