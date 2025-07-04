import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from kymatio.torch import Scattering1D
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
import warnings
import itertools
import random
from sklearn.model_selection import TimeSeriesSplit

from functions.data_processing import TimeSeriesDataset, visualize_scattering_information, extract_scattering_filters, precompute_scattering_coefficients
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
# I have now gone back to the architecture with the feeding of the scattering coefficients back into an LSTM,
# as I have introduced a sequence of scattering coefficients, which now makes sense to porcess through that type of layer

class ScatteringLSTM(nn.Module):
    def __init__(self, scattering_dim, lag_features_dim, hidden_dim, output_dim, dropout_rate, num_layers=1):
        super(ScatteringLSTM, self).__init__()
        
        self.scattering_lstm = nn.LSTM(scattering_dim, hidden_dim//2, num_layers=num_layers,
                                      batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.lstm_raw = nn.LSTM(lag_features_dim, hidden_dim, num_layers=num_layers, 
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim//2 + hidden_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 50)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(50, output_dim)

    def forward(self, scattering_x, raw_x):
        scattering_out, _ = self.scattering_lstm(scattering_x)
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

def train_and_evaluate_fold(fold_train_data, fold_val_data, model_type, params,
                           window_size, forecast_horizon, time_lags, 
                           scattering, high_energy_indices, device, step_size=4,
                           precomputed_train_scattering=None, precomputed_val_scattering=None, scattering_sequence_length=4):
    """
    Train and evaluate a single model on a CV fold
    """
    
    if model_type == 'lstm_scattering':
        train_dataset = TimeSeriesDataset(
            fold_train_data, forecast_horizon, mode='dual',
            window_size=window_size, time_lags=time_lags,
            scattering_transform=scattering, step=step_size,
            high_energy_indices=high_energy_indices,
            precomputed_scattering=precomputed_train_scattering,
            scattering_sequence_length=scattering_sequence_length
        )
        val_dataset = TimeSeriesDataset(
            fold_val_data, forecast_horizon, mode='dual',
            window_size=window_size, time_lags=time_lags,
            scattering_transform=scattering, step=1,
            high_energy_indices=high_energy_indices,
            precomputed_scattering=precomputed_val_scattering,
            scattering_sequence_length=scattering_sequence_length
        )
        
        sample_batch = next(iter(DataLoader(train_dataset, batch_size=1)))
        scattering_dim = sample_batch[0].shape[2]
        
        model = ScatteringLSTM(
            scattering_dim=scattering_dim,
            lag_features_dim=time_lags,
            hidden_dim=params['hidden_dim'],
            output_dim=forecast_horizon,
            dropout_rate=params['dropout_rate'],
            num_layers=params['num_layers']
        ).to(device)
        
    elif model_type == 'pure_lstm':
        train_dataset = TimeSeriesDataset(
            fold_train_data, forecast_horizon, mode='raw_only',
            time_lags=time_lags, step=4
        )
        val_dataset = TimeSeriesDataset(
            fold_val_data, forecast_horizon, mode='raw_only',
            time_lags=time_lags, step=1
        )
        
        model = PureLSTM(
            input_dim=time_lags,
            hidden_dim=params['hidden_dim'],
            output_dim=forecast_horizon,
            dropout_rate=params['dropout_rate'],
            num_layers=params['num_layers']
        ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Same Adam optimizer and MSE loss function as before/beyond in main training bit
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()

    # Training and evaluating here with lower number of epochs/less patience for early stopping
    # FIX REMINDER: MOVE THE SPECIFICATION OF THESE HARD CODED PARAMETERS TO THE EXECTUION SCRIPT!
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(params['epochs']):
        # Training
        model.train()
        for batch in train_loader:
            if model_type == 'lstm_scattering':
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
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'lstm_scattering':
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
    
    return best_val_loss

def rolling_window_forecast_scattering_lstm_cv(train_data, test_data, window_size, forecast_horizon, 
                                           scattering_params, lstm_params_grid, time_lags=30, 
                                           random_seed=42, step_size=4, energy_threshold=0.9,
                                           multi_step=False, forecast_steps=24, n_splits=5,
                                           num_random_starts=20, random_search_trials=15, batch_size=32, scattering_sequence_length=4):
    """
    Optimized version: Generates rolling window forecasts using Wavelet Scattering 
    Transform and LSTM, Pure LSTM, and SARIMA for large datasets

    Modified to implement time series cross validation and random starting points within the data for evaluation
    """
    
    set_random_seeds(random_seed)

    # Dataset here is almost 200k data points, so just to speed up training have limited it to 50k
    # Final thing will use more
    if len(train_data) > 100000:
        print(f"Training on last 50000 points of {len(train_data)} total points")
        train_data = train_data[-50000:]
    
    # Fitting the scaler only on the training data, for now
    scaler = RobustScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # Initializing the wavelet scattering transform, outside of the training loop
    J = int(np.log2(window_size))
    print(f"Using J={J} for scattering transform")
    
    scattering_params['J'] = J
    scattering = Scattering1D(**scattering_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    scattering.to(device)

    #Examining the scattering wavelets and filter banks at different tensor sizes (scales)

    filter_analysis = extract_scattering_filters(scattering, 1, window_size)
    filter_analysis = extract_scattering_filters(scattering, 2, window_size)
    filter_analysis = extract_scattering_filters(scattering, 0.5, window_size)

    # (moved classes out of the forecasting function for better code clarity/easier to fiddle with them)
    
    # This is like the EDA function, but is important for energy computation, which is re-used here from the graphic it creates
    # Hence why I have called the function here and not in the main execution script

    analysis_results = visualize_scattering_information(
    scattering=scattering,
    train_data=train_data,
    test_data=test_data,
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

    print("\n~~~ Pre-computing Scattering Coefficients on my training data ~~~")
    precomputed_scattering_all = precompute_scattering_coefficients(
        train_data_scaled, window_size, scattering, step=1, high_energy_indices=high_energy_indices
    )

    print("\n~~~ PHASE 1: Hyperparameter Tuning with CV ~~")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_config = None
    best_cv_score = float('inf')
    cv_results = []
    
    # Random search (i.e. just a hyperparameter sweep- grid woudl take forever)
    for trial in range(random_search_trials):
        hidden_dim = random.choice(lstm_params_grid['hidden_dims'])
        dropout_rate = random.choice(lstm_params_grid['dropout_rates'])
        learning_rate = random.choice(lstm_params_grid['learning_rates'])
        num_layers = random.choice(lstm_params_grid['num_layers'])
        
        current_params = {
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'num_layers': num_layers,
            'epochs': 30
        }
        
        print(f"\nTesting config {trial+1}/{random_search_trials}: hidden_dim={hidden_dim}, dropout={dropout_rate}, lr={learning_rate}, num_layers={num_layers}")
        
        fold_scores = {'lstm_scattering': [], 'pure_lstm': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_data_scaled)):
            print(f"  Fold {fold + 1}/{n_splits}")
            
            fold_train_data = train_data_scaled[train_idx]
            fold_val_data = train_data_scaled[val_idx]
            
            fold_train_scattering = {idx: precomputed_scattering_all[idx] 
                                   for idx in range(len(fold_train_data) - window_size + 1)}
            fold_val_scattering = {idx: precomputed_scattering_all[train_idx[0] + idx] 
                                 for idx in range(len(fold_val_data) - window_size + 1)}
            
            for model_type in ['lstm_scattering', 'pure_lstm']:
                fold_score = train_and_evaluate_fold(
                    fold_train_data, fold_val_data, model_type,
                    current_params, window_size, forecast_horizon,
                    time_lags, scattering, high_energy_indices, device, step_size=step_size,
                    precomputed_train_scattering=fold_train_scattering if model_type != 'pure_lstm' else None,
                    precomputed_val_scattering=fold_val_scattering if model_type != 'pure_lstm' else None,
                    scattering_sequence_length=scattering_sequence_length
                )
                fold_scores[model_type].append(fold_score)
        
        mean_scores = {k: np.mean(v) for k, v in fold_scores.items()}
        config_result = {
            'params': current_params,
            'cv_scores': mean_scores,
            'cv_std': {k: np.std(v) for k, v in fold_scores.items()}
        }
        cv_results.append(config_result)
        
        # Update best configuration based on the LSTM Scattering Model
        if mean_scores['lstm_scattering'] < best_cv_score:
            best_cv_score = mean_scores['lstm_scattering']
            best_config = config_result
        
        print(f"  CV Scores: LSTM+Scattering={mean_scores['lstm_scattering']:.6f}, Pure LSTM={mean_scores['pure_lstm']:.6f}")
    
    print(f"\n=== BEST CONFIGURATION ===")
    print(f"Parameters: {best_config['params']}")
    print(f"CV Score: {best_config['cv_scores']}")
    
    print("\n=== PHASE 2: Final Training on Full Data ===")
    
    # Extracting the optrimal hyperparameters combo from the random search

    best_params = best_config['params'].copy()
    best_params['epochs'] = lstm_params_grid['full_epochs']
    
    # Splitting my overall training data into training and validation sets for final 
    # model training (using the precomputed scattering coefficients ofc)

    validation_split = 0.15
    split_idx = int(len(train_data_scaled) * (1 - validation_split))
    
    final_train_data = train_data_scaled[:split_idx]
    final_val_data = train_data_scaled[split_idx:]
    
    train_scattering_dict = {idx: precomputed_scattering_all[idx] 
                           for idx in range(0, len(final_train_data) - window_size + 1, step_size)}
    val_scattering_dict = {idx: precomputed_scattering_all[split_idx + idx] 
                          for idx in range(0, len(final_val_data) - window_size + 1, 1)}
    
# Now, continuing as before, just taking these models and trianing them on the final trianing data

    train_dataset = TimeSeriesDataset(
        final_train_data, 
        forecast_horizon,
        mode='dual',
        window_size=window_size,
        time_lags=time_lags,
        scattering_transform=scattering.to(device),
        step=step_size,
        high_energy_indices=high_energy_indices,
        precomputed_scattering=train_scattering_dict,
        scattering_sequence_length=scattering_sequence_length
    )
    
    val_dataset = TimeSeriesDataset(
        final_val_data,
        forecast_horizon,
        mode='dual',
        window_size=window_size,
        time_lags=time_lags,
        scattering_transform=scattering.to(device),
        step=1,
        high_energy_indices=high_energy_indices,
        precomputed_scattering=val_scattering_dict,
        scattering_sequence_length=scattering_sequence_length
    )

    print(f"Training dataset contains {len(train_dataset)} samples with step size {step_size}")
    print(f"Validation dataset contains {len(val_dataset)} samples")

    # Creating alernative dataset for pure LSTM (without scattering coefficient adjustment)
    pure_lstm_train_dataset = TimeSeriesDataset(
        final_train_data, 
        forecast_horizon,
        mode='raw_only',
        time_lags=time_lags,
        step=step_size
    )
    
    pure_lstm_val_dataset = TimeSeriesDataset(
        final_val_data,
        forecast_horizon,
        mode='raw_only',
        time_lags=time_lags,
        step=1
    )

    batch_size = batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    pure_lstm_train_loader = DataLoader(
        pure_lstm_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    pure_lstm_val_loader = DataLoader(
        pure_lstm_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    sample_batch = next(iter(train_loader))
    sample_scattering, sample_raw, sample_target = sample_batch
    
    scattering_shape = sample_scattering.shape[2]
    print(f"Scattering shape after filtering: {scattering_shape}")
    
    # Initialize the models
    lstm_scattering_model = ScatteringLSTM(
        scattering_dim=scattering_shape,
        lag_features_dim=time_lags,
        hidden_dim=best_params['hidden_dim'],
        output_dim=forecast_horizon,
        dropout_rate=best_params['dropout_rate'],
        num_layers=best_params['num_layers']
    )
    lstm_scattering_model.to(device)
    
    pure_lstm_model = PureLSTM(
        input_dim=time_lags,
        hidden_dim=best_params['hidden_dim'],
        output_dim=forecast_horizon,
        dropout_rate=best_params['dropout_rate'],
        num_layers=best_params['num_layers']
    )
    pure_lstm_model.to(device)
    
    print(f"LSTM+Scattering Model Architecture:")
    print(f"  Hidden Dimensions: {best_params['hidden_dim']}")
    print(f"  Number of Layers: {best_params['num_layers']}")
    print(f"  Dropout Rate: {best_params['dropout_rate']}")
    print(f"  Total Parameters: {sum(p.numel() for p in lstm_scattering_model.parameters())}")
    
    print(f"\nPure LSTM Model Architecture:")
    print(f"  Hidden Dimensions: {best_params['hidden_dim']}")
    print(f"  Number of Layers: {best_params['num_layers']}")
    print(f"  Dropout Rate: {best_params['dropout_rate']}")
    print(f"  Total Parameters: {sum(p.numel() for p in pure_lstm_model.parameters())}")
    
    optimizer_scattering = optim.Adam(lstm_scattering_model.parameters(), lr=best_params['learning_rate'])
    optimizer_pure = optim.Adam(pure_lstm_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    
    lstm_scattering_start_time = time.time()
    lstm_scattering_model.train()

    early_stop_patience = 40
    best_val_loss_scattering = float('inf')
    patience_counter_scattering = 0
    
    print("\n===== Training LSTM+Scattering Model =====")
    for epoch in range(best_params['epochs']):
        epoch_train_loss = 0
        batch_count = 0
        
        lstm_scattering_model.train()
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
            
            epoch_train_loss += loss.item()
            
            # Printing batch progress at intervals of 20, since it takes forever to train and I get scared it is crashing :))
            #if batch_count % 20 == 0:
                #print(f"Epoch {epoch+1}/{best_params['epochs']}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_train_loss = epoch_train_loss / batch_count
        
        # Validation
        lstm_scattering_model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for scattering_batch, raw_batch, targets_batch in val_loader:
                val_batch_count += 1
                
                scattering_batch = scattering_batch.to(device)
                raw_batch = raw_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                outputs = lstm_scattering_model(scattering_batch, raw_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / val_batch_count
        print(f"Epoch {epoch+1}/{best_params['epochs']} complete. Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss_scattering:
            best_val_loss_scattering = avg_val_loss
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
    
    best_val_loss_pure = float('inf')
    patience_counter_pure = 0
    
    print("\n===== Training Pure LSTM Model =====")
    for epoch in range(best_params['epochs']):
        epoch_train_loss = 0
        batch_count = 0
        
        pure_lstm_model.train()
        for inputs_batch, targets_batch in pure_lstm_train_loader:
            batch_count += 1
            
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer_pure.zero_grad()
            
            outputs = pure_lstm_model(inputs_batch)
            loss = criterion(outputs, targets_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pure_lstm_model.parameters(), 1.0)
            optimizer_pure.step()
            
            epoch_train_loss += loss.item()
            
            #if batch_count % 20 == 0:
                #print(f"Epoch {epoch+1}/{best_params['epochs']}, Batch {batch_count}/{len(pure_lstm_train_loader)}, Loss: {loss.item():.6f}")
        
        avg_train_loss = epoch_train_loss / batch_count
        
        # Validation
        pure_lstm_model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for inputs_batch, targets_batch in pure_lstm_val_loader:
                val_batch_count += 1
                
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                outputs = pure_lstm_model(inputs_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / val_batch_count
        print(f"Epoch {epoch+1}/{best_params['epochs']} complete. Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss_pure:
            best_val_loss_pure = avg_val_loss
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

    print(f"\n Generating {num_random_starts} Multi-Step Recursive Forecasts")

    test_data_scaled = scaler.transform(test_data)

    min_required_length = max(window_size, time_lags) + forecast_steps
    available_starts = len(test_data) - min_required_length

    # Generate random starting points
    np.random.seed(random_seed)
    random_start_indices = np.random.choice(range(available_starts), size=num_random_starts, replace=False)
    random_start_indices.sort()

    # Store results for each random start in the following lists:

    all_lstm_scattering_sequences = []
    all_pure_lstm_sequences = []
    all_sarima_sequences = []
    all_actual_sequences = []
    all_training_times = []

    full_data_scaled = np.concatenate([train_data_scaled, test_data_scaled])
    orig_test_data = test_data.values.flatten()

    for start_idx in random_start_indices:
        print(f"\n Multi-step forecasting from starting point {start_idx}")
        
        train_end_idx = len(train_data_scaled) + start_idx - 1
        
        # Creating seperate windows for recursive forecasting
        lstm_scattering_windows = []
        for seq_idx in range(scattering_sequence_length):
            window_start = train_end_idx - (scattering_sequence_length - seq_idx) * window_size + 1
            window_end = window_start + window_size
            lstm_scattering_windows.append(full_data_scaled[window_start:window_end].flatten())
        lstm_scattering_lag = full_data_scaled[train_end_idx - time_lags + 1:train_end_idx+1].flatten()
        
        pure_lstm_lag = lstm_scattering_lag.copy()
        
        actual_sequence = test_data.iloc[start_idx:start_idx + forecast_steps].values.flatten()
        all_actual_sequences.append(actual_sequence)
        
        lstm_scattering_preds = np.zeros(forecast_steps)
        pure_lstm_preds = np.zeros(forecast_steps)
        sarima_preds = np.zeros(forecast_steps)
        
        sarima_start_time = time.time()

        orig_values = np.concatenate([
            orig_train_data,
            orig_test_data[:start_idx] if start_idx > 0 else []
        ])

        # Use a reasonable window size for SARIMA
        sarima_window_size = min(len(orig_values), 168)
        orig_window = orig_values[-sarima_window_size:]
        
        try:
            sarima_model = SARIMAX(pd.Series(orig_window), 
                                order=best_sarima_order, 
                                seasonal_order=best_sarima_seasonal)
            sarima_result = sarima_model.fit(disp=False)
            sarima_forecast = sarima_result.forecast(steps=forecast_steps)
            sarima_preds = sarima_forecast.values if hasattr(sarima_forecast, 'values') else sarima_forecast
        except Exception as e:
            print(f"SARIMA error at start_idx {start_idx}: {str(e)}")
            sarima_preds = np.repeat(orig_window[-1], forecast_steps)
        
        sarima_training_time = time.time() - sarima_start_time
        
        for step in range(forecast_steps):
            with torch.no_grad():
                scattering_sequence = []
                for window in lstm_scattering_windows:
                    window_tensor = torch.tensor(window, dtype=torch.float32).to(device)
                    window_mean = torch.mean(window_tensor)
                    window_std = torch.std(window_tensor) + 1e-8
                    window_norm = (window_tensor - window_mean) / window_std
                    
                    scattering_input = window_norm.reshape(1, 1, -1)
                    scattering_coeffs = scattering(scattering_input)
                    
                    if high_energy_indices is not None:
                        indices_tensor = torch.tensor(high_energy_indices, dtype=torch.long, device=device)
                        flat_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[0], scattering_coeffs.shape[2]).contiguous()
                        filtered_coeffs = torch.index_select(flat_coeffs, 1, indices_tensor)
                        scattering_coeffs = filtered_coeffs.reshape(filtered_coeffs.shape[1])
                    else:
                        scattering_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[2])
                    
                    scattering_sequence.append(scattering_coeffs)
                
                scattering_seq_tensor = torch.stack(scattering_sequence, dim=0).unsqueeze(0)
                
                lag_window_tensor = torch.tensor(lstm_scattering_lag, dtype=torch.float32).to(device)
                raw_input = lag_window_tensor.reshape(1, 1, -1)
                
                lstm_scattering_model.eval()
                lstm_scattering_pred_scaled = lstm_scattering_model(scattering_seq_tensor, raw_input).cpu().numpy().flatten()
                lstm_scattering_pred = scaler.inverse_transform(lstm_scattering_pred_scaled.reshape(-1, 1)).flatten()
                lstm_scattering_preds[step] = lstm_scattering_pred[0]
                
                pure_lstm_lag_tensor = torch.tensor(pure_lstm_lag, dtype=torch.float32).to(device)
                pure_lstm_input = pure_lstm_lag_tensor.reshape(1, 1, -1)
                
                pure_lstm_model.eval()
                pure_lstm_pred_scaled = pure_lstm_model(pure_lstm_input).cpu().numpy().flatten()
                pure_lstm_pred = scaler.inverse_transform(pure_lstm_pred_scaled.reshape(-1, 1)).flatten()
                pure_lstm_preds[step] = pure_lstm_pred[0]

                # Updating each window with predicted valuee to make it a true recursion
                

                for i in range(len(lstm_scattering_windows)):
                    lstm_scattering_windows[i] = np.roll(lstm_scattering_windows[i], -1)
                    lstm_scattering_windows[i][-1] = lstm_scattering_pred_scaled[0]
                
                lstm_scattering_lag = np.roll(lstm_scattering_lag, -1)
                lstm_scattering_lag[-1] = lstm_scattering_pred_scaled[0]
                
                pure_lstm_lag = np.roll(pure_lstm_lag, -1)
                pure_lstm_lag[-1] = pure_lstm_pred_scaled[0]
        
        all_lstm_scattering_sequences.append(lstm_scattering_preds)
        all_pure_lstm_sequences.append(pure_lstm_preds)
        all_sarima_sequences.append(sarima_preds)
        
        all_training_times.append({
            'lstm_scattering': lstm_scattering_training_time / num_random_starts,
            'pure_lstm': pure_lstm_training_time / num_random_starts,
            'sarima': sarima_training_time
        })
        
        print(f"Completed multi-step forecast for starting point {start_idx}")

    # Averaging the results from tacross each of the starting points and then converting to numpy arrays

    print(f"\n Averaging Results across {num_random_starts} Multi-Step Forecasts")

    all_lstm_scattering_sequences = np.array(all_lstm_scattering_sequences)
    all_pure_lstm_sequences = np.array(all_pure_lstm_sequences)
    all_sarima_sequences = np.array(all_sarima_sequences)
    all_actual_sequences = np.array(all_actual_sequences)

    training_times = {
        'lstm_scattering': [lstm_scattering_training_time],
        'pure_lstm': [pure_lstm_training_time],
        'sarima': [np.mean([t['sarima'] for t in all_training_times])]
    }

    weight_analysis = analyze_model_weights(lstm_scattering_model, best_params)

    return (all_lstm_scattering_sequences, all_pure_lstm_sequences, 
            all_actual_sequences, all_sarima_sequences, training_times, lstm_scattering_model, 
            pure_lstm_model, 
            random_start_indices, best_config)