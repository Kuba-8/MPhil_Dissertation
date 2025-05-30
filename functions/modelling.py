import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from kymatio.torch import Scattering1D
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import warnings
import itertools
import random
from sklearn.model_selection import TimeSeriesSplit

from functions.data_processing import TimeSeriesDataset, visualize_scattering_information
from functions.evaluation import analyze_model_weights

def set_random_seeds(seed_value=42):
    """
    Setting seeds across all random number generators,
    to minimize, as much as I can, replicability issues
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed_value}")

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

def rolling_window_forecast_scattering_lstm_cv(train_data, test_data, window_size, forecast_horizon, 
                                           scattering_params, lstm_params, time_lags=30, 
                                           random_seed=42, step_size=4, energy_threshold=0.9,
                                           multi_step=False, forecast_steps=24, n_splits=5,
                                           num_random_starts=20):
    """
    Optimized version: Generates rolling window forecasts using Wavelet Scattering 
    Transform and LSTM, Pure LSTM, Scattering-Only LSTM, and SARIMA for large datasets

    Modified to implement time series cross validation and random starting points within the data for evaluation
    """
    
    set_random_seeds(GLOBAL_SEED)
    
    # Fitting the scaler only on the training data, for now
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    
    # Dataset here is almost 200k data points, so just to speed up training have limited it to 8k
    # Final thing will use more
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

    # (moved classes out of the forecasting function for better code clarity/easier to fiddle with them)
    
    # This is like the EDA function, but is important for energy computation, which is re-used here from the graphic it creates
    # Hence why I have called the function here and not in the main execution script

    analysis_results = visualize_scattering_information(
        scattering=scattering,
        data=train_data,
        scattering_params=scattering_params,
        window_size=window_size,
        num_samples=5
    )
    
    # Extraccting my high energy coefficients (to not overparameterize the model),
    # to the specified threshold (hyperparameter in main executionn script)

    high_energy_indices = None
    if energy_threshold < 1.0:
        idx_threshold = analysis_results['idx_90'] if energy_threshold == 0.9 else analysis_results['idx_99']
        sorted_indices = np.argsort(-analysis_results['average_magnitudes'])
        high_energy_indices = np.ascontiguousarray(sorted_indices[:idx_threshold])
        print(f"Using {len(high_energy_indices)} out of {len(sorted_indices)} scattering coefficients ({len(high_energy_indices)/len(sorted_indices):.1%})")
        print(f"High energy indices shape: {high_energy_indices.shape}, dtype: {high_energy_indices.dtype}")
        print(f"First few indices: {high_energy_indices[:5]}")

    # Adding time-series cross validation here to stop model concentrating on a particular period
    # (SIGNIFICANT issue I encountered)

    print(f"\nTime-Series CV with {n_splits} splits")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'lstm_scattering': [],
        'pure_lstm': [],
        'scattering_only': []
    }
    
    # Performing cross-validation on me training data, per fold
    # Not particularly complex- much of the same code as below just stuck in a for-loop
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_data_scaled)):
        print(f"\n---Cross-Validation Fold {fold + 1}/{n_splits} ---")
        print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
        
        fold_train_data = train_data_scaled[train_idx]
        fold_val_data = train_data_scaled[val_idx]

        # Creating a training and validation set for this particular fold
        
        fold_train_dataset = TimeSeriesDataset(
            fold_train_data, 
            forecast_horizon,
            mode='dual',
            window_size=window_size, 
            time_lags=time_lags,
            scattering_transform=scattering.to(device),
            step=step_size,
            high_energy_indices=high_energy_indices
        )
        
        fold_val_dataset = TimeSeriesDataset(
            fold_val_data,
            forecast_horizon,
            mode='dual',
            window_size=window_size, 
            time_lags=time_lags,
            scattering_transform=scattering.to(device),
            step=1,
            high_energy_indices=high_energy_indices
        )
        
        # Batch dataloaders for parallel processing hereee
        batch_size = 32
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        sample_batch = next(iter(fold_train_loader))
        sample_scattering, sample_raw, sample_target = sample_batch
        scattering_shape = sample_scattering.shape[1]
        
        # Initializing the models (again for this fold within the CV loop)
        fold_lstm_scattering_model = ScatteringLSTM(
            scattering_dim=scattering_shape,
            lag_features_dim=time_lags,
            hidden_dim=lstm_params['hidden_dim'],
            output_dim=forecast_horizon,
            dropout_rate=lstm_params['dropout_rate'],
            num_layers=lstm_params['num_layers']
        ).to(device)
        
        fold_pure_lstm_model = PureLSTM(
            input_dim=time_lags,
            hidden_dim=lstm_params['hidden_dim'],
            output_dim=forecast_horizon,
            dropout_rate=lstm_params['dropout_rate'],
            num_layers=lstm_params['num_layers']
        ).to(device)
        
        fold_scattering_only_model = ScatteringOnlyLSTM(
            scattering_dim=scattering_shape,
            hidden_dim=lstm_params['hidden_dim'],
            output_dim=forecast_horizon,
            dropout_rate=lstm_params['dropout_rate'],
            num_layers=lstm_params['num_layers']
        ).to(device)
        
        # In training and validating each model for this fold, I think it is important to note here
        # that the 3rd term of each of the list of tuples determines the relevant inputs
        # e.g., for the first one, indicates teh dual input of scattering coefficients 
        # and raw data time lags into the model
        # (2 separate features)

        models_and_configs = [
            (fold_lstm_scattering_model, 'lstm_scattering', 'dual'),
            (fold_pure_lstm_model, 'pure_lstm', 'raw_only'),
            (fold_scattering_only_model, 'scattering_only', 'scattering_only')
        ]

        for model, model_name, dataset_mode in models_and_configs:
            print(f"\nTraining {model_name} for fold {fold + 1}")
            
            if dataset_mode == 'dual':
                train_loader = fold_train_loader
                val_loader = fold_val_loader
            else:
                fold_mode_train = TimeSeriesDataset(
                    fold_train_data, 
                    forecast_horizon,
                    mode=dataset_mode,
                    window_size=window_size if dataset_mode == 'scattering_only' else None,
                    time_lags=time_lags if dataset_mode == 'raw_only' else None,
                    scattering_transform=scattering.to(device) if dataset_mode == 'scattering_only' else None,
                    step=step_size,
                    high_energy_indices=high_energy_indices if dataset_mode == 'scattering_only' else None
                )
                
                fold_mode_val = TimeSeriesDataset(
                    fold_val_data,
                    forecast_horizon,
                    mode=dataset_mode,
                    window_size=window_size if dataset_mode == 'scattering_only' else None,
                    time_lags=time_lags if dataset_mode == 'raw_only' else None,
                    scattering_transform=scattering.to(device) if dataset_mode == 'scattering_only' else None,
                    step=1,
                    high_energy_indices=high_energy_indices if dataset_mode == 'scattering_only' else None
                )
                    
                train_loader = DataLoader(fold_mode_train, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(fold_mode_val, batch_size=batch_size, shuffle=False, num_workers=0)
            
            
            # Same Adam optimizer and MSE loss function as before/beyond in main training bit

            optimizer = optim.Adam(model.parameters(), lr=lstm_params['learning_rate'])
            criterion = nn.MSELoss()
            
            # Training and evaluating here with lower number of epochs/less patience for early stopping
            # FIX REMINDER: MOVE THE SPECIFICATION OF THESE HARD CODED PARAMETERS TO THE EXECTUION SCRIPT!

            best_val_loss = float('inf')
            patience_counter = 0
            patience = 15
            
            cv_epochs = min(lstm_params['epochs'] // 2, 50)
            
            # Another nested for-loop... literally just training and evaluating on validation set of fold

            for epoch in range(cv_epochs):
                # Training
                model.train()
                train_loss = 0
                for batch in train_loader:
                    if dataset_mode == 'dual':
                        scattering_input, raw_input, targets = [b.to(device) for b in batch]
                        outputs = model(scattering_input, raw_input)
                    else:
                        inputs, targets = [b.to(device) for b in batch]
                        outputs = model(inputs)
                    
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if dataset_mode == 'dual':
                            scattering_input, raw_input, targets = [b.to(device) for b in batch]
                            outputs = model(scattering_input, raw_input)
                        else:
                            inputs, targets = [b.to(device) for b in batch]
                            outputs = model(inputs)
                        
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping to stop overfitting
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Storing modelling result for this fold
            cv_scores[model_name].append(best_val_loss)
            print(f"{model_name} fold {fold + 1} validation loss: {best_val_loss:.6f}")
    
    # Printing off the cross-validation results, as a bit of a debugging step
    print("\nCV Results")
    for model_name, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{model_name}: {mean_score:.6f} ± {std_score:.6f} (CV Loss)")
    
    # Now, continuing as before, just taking these models and trianing them on the final trianing data
    
    train_dataset = TimeSeriesDataset(
        train_data_scaled, 
        forecast_horizon,
        mode='dual',
        window_size=window_size,
        time_lags=time_lags,
        scattering_transform=scattering.to(device),
        step=step_size,
        high_energy_indices=high_energy_indices
    )

    print(f"Training dataset contains {len(train_dataset)} samples with step size {step_size}")

    # Creating alernative dataset for pure LSTM (without scattering coefficient adjustment)
    pure_lstm_dataset = TimeSeriesDataset(
        train_data_scaled, 
        forecast_horizon,
        mode='raw_only',
        time_lags=time_lags,
        step=step_size
    )

    # Creating an alternative dataset for the scattering-only LSTM model 
    # (no raw data points, only scattering coefficients)
    scattering_only_dataset = TimeSeriesDataset(
        train_data_scaled,
        forecast_horizon,
        mode='scattering_only',
        window_size=window_size,
        scattering_transform=scattering.to(device),
        step=step_size,
        high_energy_indices=high_energy_indices
    )

    batch_size = 32
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
    sample_scattering, sample_raw, sample_target = sample_batch
    
    scattering_shape = sample_scattering.shape[1]
    print(f"Scattering shape after filtering: {scattering_shape}")
    
    scattering_only_sample = next(iter(scattering_only_loader))
    scattering_only_input, _ = scattering_only_sample
    
    # Initialize the models
    lstm_scattering_model = ScatteringLSTM(
        scattering_dim=scattering_shape,
        lag_features_dim=time_lags,
        hidden_dim=lstm_params['hidden_dim'],
        output_dim=forecast_horizon,
        dropout_rate=lstm_params['dropout_rate'],
        num_layers=lstm_params['num_layers']
    )
    lstm_scattering_model.to(device)
    
    pure_lstm_model = PureLSTM(
        input_dim=time_lags,
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
    optimizer_scattering_only = optim.Adam(scattering_only_model.parameters(), lr=lstm_params['learning_rate'])
    criterion = nn.MSELoss()
    
    lstm_scattering_start_time = time.time()
    lstm_scattering_model.train()

    early_stop_patience = 40
    early_stop_patience_scattering_only = 10
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
            print(f"Early stopping patience: {patience_counter_scattering_only}/{early_stop_patience_scattering_only}")
            if patience_counter_scattering_only >= early_stop_patience_scattering_only:
                print("Early stopping triggered!")
                break
    
    # Loading the best pure scattering model
    scattering_only_model.load_state_dict(torch.load('best_energy_forecast_scattering_only_model.pt'))
    
    scattering_only_training_time = time.time() - scattering_only_start_time
    print(f"Scattering-Only LSTM training completed in {scattering_only_training_time:.2f} seconds")
    
    # Updated section here!
    # Put in a SARIMA function using AIC score minimisation, instead of just using the random ARIMA
    # (as I said I would)

    print("\n Finding Optimal SARIMA Parameters")
    
    # (Going back to non-normalized data too)
    orig_train_data = train_data.values.flatten()
    
    # Searching 2000 data points over daily (24hour) seasonality for a low order SARIMA
    # (p d and q are inputted fairly low just to keep small search space)
    # (forecast converges to an estimated mean fairly quickly anyway, so does not really matter imho)

    best_sarima_order, best_sarima_seasonal = find_optimal_sarima(
        orig_train_data[-min(len(orig_train_data), 2000):],
        max_p=2, max_d=2, max_q=2, 
        max_P=1, max_D=1, max_Q=1, 
        seasonal_period=24
    )
    
    # Now have also introduced random forecast starting points within the test data
    # (before, was just really doing it for the first few points, this is far more robust)
    print(f"\n Forecasting from {num_random_starts} Random Starting Points")
    
    test_data_scaled = scaler.transform(test_data)
    
    min_required_length = max(window_size, time_lags) + forecast_horizon
    available_starts = len(test_data) - min_required_length
    
    # Generate random starting points
    np.random.seed(GLOBAL_SEED)
    random_start_indices = np.random.choice(range(available_starts), size=num_random_starts, replace=False)
    random_start_indices.sort()
    
    # Store results for each random start in the following lists:

    all_lstm_scattering_forecasts = []
    all_pure_lstm_forecasts = []
    all_scattering_only_forecasts = []
    all_sarima_forecasts = []
    all_actual_values = []
    all_training_times = []
    
    full_data_scaled = np.concatenate([train_data_scaled, test_data_scaled])
    orig_test_data = test_data.values.flatten()
    
    for start_idx in random_start_indices:
        print(f"\n Forecasting from starting point {start_idx}")
        train_end_idx = len(train_data_scaled) + start_idx - 1
        window_start_idx = train_end_idx - window_size + 1
        current_window = full_data_scaled[window_start_idx:train_end_idx+1].flatten()
        
        # To incorporate the time lags as a feature, have to get time lag window here
        lag_end_idx = train_end_idx
        lag_start_idx = lag_end_idx - time_lags + 1
        current_lag_window = full_data_scaled[lag_start_idx:lag_end_idx+1].flatten()
        
        actual = test_data.iloc[start_idx + forecast_horizon - 1].values.flatten()
        all_actual_values.append(actual[0])
        
        with torch.no_grad():
            window_tensor = torch.tensor(current_window, dtype=torch.float32).to(device)
            lag_window_tensor = torch.tensor(current_lag_window, dtype=torch.float32).to(device)
            
            # Normalize window before scattering transform (consistent with visualization function)
            # This bit I am not sure about, might be worth changign
            window_norm = (window_tensor - torch.mean(window_tensor)) / (torch.std(window_tensor) + 1e-8)
            
            # Reshaping for scattering transform
            scattering_input = window_norm.reshape(1, 1, -1)
            scattering_coeffs = scattering(scattering_input)
            
            # Filtering the scattering coefficients for dimensionalty reduction
            if high_energy_indices is not None:
                indices_tensor = torch.tensor(high_energy_indices, dtype=torch.long, device=device)
                flat_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[0], scattering_coeffs.shape[2]).contiguous()
                filtered_coeffs = torch.index_select(flat_coeffs, 1, indices_tensor)
                scattering_coef_input = filtered_coeffs.reshape(1, filtered_coeffs.shape[1], 1)
            else:
                scattering_coef_input = scattering_coeffs.reshape(1, scattering_coeffs.shape[2], 1)
                
            # Reshaping again, for the LSTM and scattering input (differentlly though)
            raw_input = lag_window_tensor.reshape(1, 1, -1)
            
            # Doing the forecast with the dual input
            lstm_scattering_model.eval()
            lstm_scattering_pred_scaled = lstm_scattering_model(scattering_coef_input, raw_input).cpu().numpy().flatten()
            
            lstm_scattering_pred = scaler.inverse_transform(lstm_scattering_pred_scaled.reshape(-1, 1)).flatten()
            all_lstm_scattering_forecasts.append(lstm_scattering_pred[0])
            
            # Doing the pure LSTM forecast
            pure_lstm_model.eval()
            pure_lstm_input = lag_window_tensor.reshape(1, 1, -1)
            pure_lstm_pred_scaled = pure_lstm_model(pure_lstm_input).cpu().numpy().flatten()
            
            pure_lstm_pred = scaler.inverse_transform(pure_lstm_pred_scaled.reshape(-1, 1)).flatten()
            all_pure_lstm_forecasts.append(pure_lstm_pred[0])
            
            # Doing the pure scattering LSTM forecast
            scattering_only_model.eval()
            scattering_only_pred_scaled = scattering_only_model(scattering_coef_input).cpu().numpy().flatten()
            
            scattering_only_pred = scaler.inverse_transform(scattering_only_pred_scaled.reshape(-1, 1)).flatten()
            all_scattering_only_forecasts.append(scattering_only_pred[0])
        
        # SARIMA forecast bit
        sarima_start_time = time.time()
        
        orig_values = np.concatenate([
            orig_train_data,
            orig_test_data[:start_idx] if start_idx > 0 else []
        ])
        
        # Use a reasonable window size for SARIMA
        sarima_window_size = min(len(orig_values), 168)  # Use last week of data
        orig_window = orig_values[-sarima_window_size:]
        
        try:
            sarima_model = SARIMAX(pd.Series(orig_window), 
                                 order=best_sarima_order, 
                                 seasonal_order=best_sarima_seasonal)
            sarima_result = sarima_model.fit(disp=False)
            sarima_pred = sarima_result.forecast(steps=forecast_horizon)
            all_sarima_forecasts.append(sarima_pred.iloc[0] if hasattr(sarima_pred, 'iloc') else sarima_pred[0])
        except Exception as e:
            print(f"SARIMA error at start_idx {start_idx}: {str(e)}")
            all_sarima_forecasts.append(orig_window[-1])
        
        sarima_training_time = time.time() - sarima_start_time
        all_training_times.append({
            'lstm_scattering': lstm_scattering_training_time / num_random_starts,
            'scattering_only': scattering_only_training_time / num_random_starts,
            'pure_lstm': pure_lstm_training_time / num_random_starts,
            'sarima': sarima_training_time
        })
        
        print(f"Completed forecast from start {start_idx}: LSTM+Scattering: {all_lstm_scattering_forecasts[-1]:.2f}, "
              f"Scattering-Only: {all_scattering_only_forecasts[-1]:.2f}, Pure LSTM: {all_pure_lstm_forecasts[-1]:.2f}, "
              f"SARIMA: {all_sarima_forecasts[-1]:.2f}, Actual: {actual[0]:.2f}")
    
    # Averaging the results from tacross each of the starting points and then converting to numpy arrays

    print(f"\n Averaging Results across {num_random_starts} Random Starting Points")
    
    lstm_scattering_forecasts = np.array(all_lstm_scattering_forecasts)
    pure_lstm_forecasts = np.array(all_pure_lstm_forecasts)
    scattering_only_forecasts = np.array(all_scattering_only_forecasts)
    sarima_forecasts = np.array(all_sarima_forecasts)
    actual_values = np.array(all_actual_values)
    
    print(f"LSTM+Scattering forecasts: Mean={np.mean(lstm_scattering_forecasts):.2f}, Std={np.std(lstm_scattering_forecasts):.2f}")
    print(f"Scattering-Only forecasts: Mean={np.mean(scattering_only_forecasts):.2f}, Std={np.std(scattering_only_forecasts):.2f}")
    print(f"Pure LSTM forecasts: Mean={np.mean(pure_lstm_forecasts):.2f}, Std={np.std(pure_lstm_forecasts):.2f}")
    print(f"SARIMA forecasts: Mean={np.mean(sarima_forecasts):.2f}, Std={np.std(sarima_forecasts):.2f}")
    print(f"Actual values: Mean={np.mean(actual_values):.2f}, Std={np.std(actual_values):.2f}")
    
    training_times = {
        'lstm_scattering': [lstm_scattering_training_time],
        'scattering_only': [scattering_only_training_time], 
        'pure_lstm': [pure_lstm_training_time],
        'sarima': [np.mean([t['sarima'] for t in all_training_times])]
    }
    
    weight_analysis = analyze_model_weights(lstm_scattering_model, lstm_params)
    
    return (lstm_scattering_forecasts, pure_lstm_forecasts, scattering_only_forecasts, 
            actual_values, sarima_forecasts, training_times, lstm_scattering_model, 
            pure_lstm_model, scattering_only_model)