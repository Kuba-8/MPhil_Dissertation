import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf
from torch.utils.data import Dataset

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
    def __init__(self, data, forecast_horizon, mode='dual', 
                 window_size=None, time_lags=30, scattering_transform=None, 
                 step=1, energy_threshold=0.9, high_energy_indices=None,
                 precomputed_scattering=None):
        """
        Initialize the unified dataset
        
        """
        self.data = data
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        self.window_size = window_size
        self.time_lags = time_lags
        self.scattering = scattering_transform
        self.step = step
        self.energy_threshold = energy_threshold
        self.high_energy_indices = high_energy_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precomputed_scattering = precomputed_scattering
        
        if mode in ['dual', 'scattering_only']:
            if window_size is None or (scattering_transform is None and precomputed_scattering is None):
                raise ValueError(f"window_size and (scattering_transform or precomputed_scattering) required for mode '{mode}'")
        
        if mode in ['dual', 'raw_only']:
            if time_lags is None:
                raise ValueError(f"time_lags required for mode '{mode}'")
        
        if mode == 'dual':
            max_lookback = max(window_size, time_lags)
        elif mode == 'raw_only':
            max_lookback = time_lags
        elif mode == 'scattering_only':
            max_lookback = window_size
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'dual', 'raw_only', or 'scattering_only'")
        
        self.indices = list(range(0, len(data) - max_lookback - forecast_horizon + 1, step))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        scattering_coeffs = None
        lag_window_tensor = None
        
        if self.mode == 'dual':
            target_start = start_idx + self.window_size
        elif self.mode == 'raw_only':
            target_start = start_idx + self.time_lags
        elif self.mode == 'scattering_only':
            target_start = start_idx + self.window_size
            
        target = self.data[target_start:target_start + self.forecast_horizon].flatten()
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        if self.mode in ['dual', 'scattering_only']:
            if self.precomputed_scattering is not None and start_idx in self.precomputed_scattering:
                scattering_coeffs = self.precomputed_scattering[start_idx].to(self.device)
            else:
                scatter_window = self.data[start_idx:start_idx + self.window_size].flatten()
                scatter_window_tensor = torch.tensor(scatter_window, dtype=torch.float32)
                
                with torch.no_grad():
                    scattering_input = scatter_window_tensor.unsqueeze(0).unsqueeze(0)
                    scattering_coeffs = self.scattering(scattering_input)
                    
                    if self.high_energy_indices is not None:
                        indices_tensor = torch.tensor(self.high_energy_indices, dtype=torch.long, device=self.device)
                        flat_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[0], scattering_coeffs.shape[2]).contiguous()
                        filtered_coeffs = torch.index_select(flat_coeffs, 1, indices_tensor)
                        scattering_coeffs = filtered_coeffs.reshape(filtered_coeffs.shape[1], 1)
                    else:
                        scattering_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[2], 1)
        
        if self.mode in ['dual', 'raw_only']:
            if self.mode == 'dual':
                lag_start = start_idx + self.window_size - self.time_lags
                lag_end = start_idx + self.window_size
            else:
                lag_start = start_idx
                lag_end = start_idx + self.time_lags
                
            lag_window = self.data[lag_start:lag_end].flatten()
            lag_window_tensor = torch.tensor(lag_window, dtype=torch.float32)
            lag_window_tensor = lag_window_tensor.reshape(1, -1)
        
        if self.mode == 'dual':
            return scattering_coeffs, lag_window_tensor, target_tensor
        elif self.mode == 'raw_only':
            return lag_window_tensor, target_tensor
        elif self.mode == 'scattering_only':
            return scattering_coeffs, target_tensor

def precompute_scattering_coefficients(data, window_size, scattering_transform, step=1, high_energy_indices=None):
    """
    Simple function that pre-computes the scattering coefficients for each and every windows in the dataset
    (Hugely impactful in terms of memory and processing time)
   
    """

    try:
        device = next(scattering_transform.parameters()).device
        print(f"Using device: {device}")
    except (StopIteration, AttributeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using default device: {device}")
        scattering_transform = scattering_transform.to(device)
    
    scattering_coeffs_dict = {}
    
    indices = list(range(0, len(data) - window_size + 1, step))
    
    print(f"...pre-computing scattering coefficients for {len(indices)} windows...")
    
    for i, start_idx in enumerate(indices):
        window = data[start_idx:start_idx + window_size].flatten()
        window_tensor = torch.tensor(window, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            scattering_input = window_tensor.unsqueeze(0).unsqueeze(0)
            scattering_coeffs = scattering_transform(scattering_input)
            
            if high_energy_indices is not None:
                indices_tensor = torch.tensor(high_energy_indices, dtype=torch.long, device=device)
                flat_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[0], scattering_coeffs.shape[2]).contiguous()
                filtered_coeffs = torch.index_select(flat_coeffs, 1, indices_tensor)
                scattering_coeffs = filtered_coeffs.reshape(filtered_coeffs.shape[1], 1)
            else:
                scattering_coeffs = scattering_coeffs.reshape(scattering_coeffs.shape[2], 1)
            
            scattering_coeffs_dict[start_idx] = scattering_coeffs.cpu()
        
        if (i + 1) % 100 == 0:
            print(f"  Computed {i + 1}/{len(indices)} windows")
    
    print(f"pre-computation complete. Stored {len(scattering_coeffs_dict)} coefficient sets.")
    return scattering_coeffs_dict

def visualize_scattering_information(scattering, train_data, test_data, scattering_params, window_size, sample_indices=None, num_samples=5):
    """
    A bit of an EDA (exploratory data ananlysis) function for graphing the data,
    alongside the information extracted by the wavelet scattering transform
    (e.g. energy content of the coefficients)
    
    Visualize the information content extracted by wavelet scattering transforms.
    
    """
    # Combining the train and test data to allow for complete EDA 

    complete_data = pd.concat([train_data, test_data])
    train_end_idx = len(train_data)
    
    if isinstance(complete_data, pd.DataFrame):
        data_values = complete_data.values
        time_index = complete_data.index
    else:
        data_values = complete_data
        time_index = range(len(complete_data.flatten()))
    
    # Does not want to use my GPU for some reason, so set a backup
    try:
        device = next(scattering.parameters()).device
        print(f"Using device: {device}")
    except (StopIteration, AttributeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using default device: {device}")
    
    # Some simple, raw data plots to begin with

    plt.figure(figsize=(24, 22))

    # Highlighting the train-test split for the time series

    plt.subplot(5, 2, 1)
    data_flat = data_values.flatten()
    
    train_data_flat = train_data.values.flatten()
    test_data_flat = test_data.values.flatten()
    
    plt.plot(train_data.index, train_data_flat, color='blue', alpha=0.7, linewidth=0.8, label='Training Data')
    plt.plot(test_data.index, test_data_flat, color='orange', alpha=0.7, linewidth=0.8, label='Test Data')
    plt.title(f'Complete Dataset - Train: {len(train_data_flat)} pts, Test: {len(test_data_flat)} pts', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Energy Load (kWh)', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # One years of raw tiem series data, for two random years (e.g. 2017)

    plt.subplot(5, 2, 2)
    start_2016 = pd.to_datetime('2017-01-01')
    end_2018 = pd.to_datetime('2018-01-01')

    mask_2016_2018 = (complete_data.index >= start_2016) & (complete_data.index < end_2018)
    subset_data = complete_data[mask_2016_2018]
    plt.plot(subset_data.index, subset_data.values.flatten())
    plt.title(f'Energy Data 2017-2018 ({len(subset_data)} points)', fontsize=14)
    plt.xlabel('Date/Time', fontsize=12)
    plt.ylabel('Energy load (kWh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # ACF plot, using first 5000 points and seasonally adjusted series (with CI's)
    # (plotted it before, and ACF just went up and down like the signal, which is unhelpful)
    # So, I am basically taking the 24th difference of the data to remove daily seasonality 
    # As per standard practice 

    plt.subplot(5, 2, 3)
    acf_data = data_flat[:min(5000, len(data_flat))]
    
    acf_data_seasonal = acf_data[24:] - acf_data[:-24]
    acf_series = acf_data_seasonal
    series_description = "Seasonally Adjusted for daily seasonality"
    acf_values = acf(acf_series, nlags=min(50, len(acf_series)//4), fft=True)
    
    plt.plot(range(len(acf_values)), acf_values, 'b-', alpha=0.8, linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    n = len(acf_series)
    confidence_interval = 1.96 / np.sqrt(n)
    plt.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.5, label='95% Confidence')
    plt.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f'ACF - {series_description} Series', fontsize=14)
    plt.xlabel('Lag (hours)', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Changing the month on month long memory graph to superimpose months more accurately
    # by matching days of the week, as this is a strong pattern within my data 
    # (differential energy consumption on weekends in comparison to the weekdays)
    # Hence, the month length is much shorter to allow matching

    plt.subplot(5, 2, 4)

    month_length = 24 * 25
    colors_months = plt.cm.viridis(np.linspace(0, 1, 3))
    month_names = ['June 2016', 'July 2016', 'August 2016']

    target_weekday = 0
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    aligned_data = []
    actual_start_dates = []
    successful_months = []

    print(f"Using month_length = {month_length} points ({month_length/24:.0f} days)")

    for i, month_num in enumerate([6, 7, 8]):
        month_mask = (complete_data.index.year == 2016) & (complete_data.index.month == month_num)
        month_data_available = complete_data[month_mask]
        
        if len(month_data_available) == 0:
            continue
        
        month_start = month_data_available.index[0]
        current_weekday = month_start.weekday()
        
        if current_weekday <= target_weekday:
            days_to_skip = target_weekday - current_weekday
        else:
            days_to_skip = 7 - (current_weekday - target_weekday)
        
        aligned_start = month_start + pd.Timedelta(days=days_to_skip)
        
        if aligned_start in month_data_available.index:
            data_from_aligned = month_data_available.loc[aligned_start:]
            if len(data_from_aligned) >= month_length:
                month_data = data_from_aligned.iloc[:month_length].values.flatten()
                actual_start_dates.append(aligned_start)
                aligned_data.append(month_data)
                successful_months.append(i)
                
                plt.plot(range(len(month_data)), month_data, alpha=0.8, 
                        color=colors_months[i], 
                        label=f'{month_names[i]} (from {aligned_start.strftime("%b %d")})', 
                        linewidth=1.5)
                
                print(f"{month_names[i]}: {len(month_data)} points from {aligned_start.strftime('%A, %B %d')}")

    plt.title('Superimposed Summer Months from 2016', fontsize=12)
    plt.xlabel('Hours from first Monday of the month', fontsize=12)
    plt.ylabel('Energy load in kWh', fontsize=12)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Superimposed 3 consecutive years with hourly data (not daily averaged)

    plt.subplot(5, 2, 5)
    years_to_plot = [2012, 2013, 2014]
    colors_years = plt.cm.plasma(np.linspace(0, 1, 5))
    
    for i, year in enumerate(years_to_plot):
        year_mask = complete_data.index.year == year
        if year_mask.any():
            year_data = complete_data[year_mask].values.flatten()
            display_length = min(len(year_data), 8760)
            if display_length > 0:
                hour_range = range(display_length)
                plt.plot(hour_range, year_data[:display_length], alpha=0.8, 
                        color=colors_years[i], label=f'Year {year}', linewidth=1.0)
    
    plt.title(f'Superimposed Energy Load for 2012, 2013 and 2014', fontsize=14)
    plt.xlabel('Hours within a Year', fontsize=12)
    plt.ylabel('Energy Load (kWh)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Testing out the scattering transform on the first few "windows" of the training data
    # To avoid any semblance of data leakage

    if sample_indices is None:
        max_idx = len(train_data.values) - window_size
        sample_indices = np.linspace(0, max_idx, num_samples).astype(int)
    
    print(f"analyzing {len(sample_indices)} samples with a Window size of {window_size}")
    
    samples = []
    scattering_outputs = []
    
    plt.subplot(5, 2, 6)
    
    for i, idx in enumerate(sample_indices):
        sample = train_data.values[idx:idx + window_size].flatten()
        samples.append(sample)
        
        sample_norm = (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
        
        sample_tensor = torch.tensor(sample_norm.reshape(1, 1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            scatter_output = scattering(sample_tensor)
            scattering_outputs.append(scatter_output.cpu().numpy())
        if i < 5:
            color = plt.cm.viridis(i / 4)
            plt.plot(sample_norm, alpha=0.7, label=f'Sample {i+1}', color=color)
    
    plt.title('Sample Windows for scattering Analysis')
    plt.xlabel('time points', fontsize=12)
    
    plt.ylabel('Normalized values', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Showing daily seasonality for the first month (4 weeks) of data

    plt.subplot(5, 2, 7)
    sample_size = min(24*7*4, len(data_flat))
    seasonal_data = data_flat[:sample_size]
    
    days_to_show = min(7, len(seasonal_data) // 24)
    colors_days = plt.cm.inferno(np.linspace(0, 1, days_to_show))
    
    for day in range(days_to_show):
        day_start = day * 24
        day_end = day_start + 24
        day_data = seasonal_data[day_start:day_end]
        plt.plot(range(24), day_data, alpha=0.8, 
                color=colors_days[day], label=f'Day {day+1}')
    
    plt.title('Daily patterns- first week of the dataset)', fontsize=12)
    plt.xlabel('Hour of the day', fontsize=12)
    plt.ylabel('Energy Load in kWh', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Energy load distribution histogram
    # This is just a simple histogram of the time series values, in order to observe any outliers/skewness etc etc

    plt.subplot(5, 2, 8)
    plt.hist(data_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Energy Load Distribution', fontsize=14)
    plt.xlabel('Energy Load (kWh)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    mean_val = np.mean(data_flat)
    median_val = np.median(data_flat)
    std_val = np.std(data_flat)
    
    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
    plt.legend(fontsize=9)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Now, starting a second, separate figure for a few wavelet scattering analysis graphs, 
    # that will be important in order to explain/apply my proposed WST-LSTM method

    plt.figure(figsize=(16, 12))
    
    # Taking a couple of the scattering tranforms computed for the sample windows graph above,
    # and averaging the magnitudes of the generated coefficients

    scattering_array = np.array([s.flatten() for s in scattering_outputs])
    avg_magnitudes = np.mean(np.abs(scattering_array), axis=0)
    
    # Just plotting the avg magnitude of each coefficient

    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(avg_magnitudes)), avg_magnitudes, 'o-', alpha=0.7)
    plt.yscale('log')
    plt.title('Scattering Coefficient Magnitude, on a Log Scale', fontsize=14)
    plt.xlabel('coefficient index', fontsize=12)
    plt.ylabel('average magnitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Cumulative energy is calculated by the formula below, 
    # derived from a defining feature of the WST, 
    # that coefficients contain 100% of the energy of the original signal

    cumulative_energy = np.cumsum(avg_magnitudes**2) / np.sum(avg_magnitudes**2)
    
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(cumulative_energy)), cumulative_energy, 'r-')
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.8)
    plt.axhline(y=0.99, color='gray', linestyle='--', alpha=0.8)
    
    # Finding indices where we capture 90% and 99% of energy, 
    
    # (useful for computational efficiency)

    idx_90 = np.argmax(cumulative_energy >= 0.9)
    idx_99 = np.argmax(cumulative_energy >= 0.99)
    
    plt.scatter([idx_90], [cumulative_energy[idx_90]], color='blue', s=50, 
                label=f'90% Energy: {idx_90} coeffs ({idx_90/len(cumulative_energy):.1%})')
    plt.scatter([idx_99], [cumulative_energy[idx_99]], color='green', s=50, 
                label=f'99% Energy: {idx_99} coeffs ({idx_99/len(cumulative_energy):.1%})')
    
    plt.title('Cumulative Energy Distribution', fontsize=14)
    plt.xlabel('Number of Coefficients', fontsize=12)
    plt.ylabel('Cumulative energy ratio', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Temporal stability analysis at specific time intervals
    # Essentially assessing the translation invariance of the WST representation,
    # beyond its averaging (invariance) scale
    # For example, for a window of 128 hours, the coefficients should be constant for a 24 hour shift
    # but not for a 2 month shift
    # (limiting the analysis to the first 100 coefficients for clarity)

    plt.subplot(2, 1, 2)
    
    train_values = train_data.values.flatten()
    
    time_intervals = [
        (0, "Start (t=0)"),
        (24, "1 Day Later (t+24h)"),
        (168, "1 Week Later (t+168h)"),
        (336, "2 Weeks Later (t+336h)"),
        (720, "1 Month Later (t+720h)"),
        (1440, "2 Months Later (t+1440h)"),
        (4320, "6 Months Later (t+4320h)"),
        (8760, "1 Year Later (t+8760h)"),
        (17520, "2 Years Later (t+17520h)")
    ]
    
    temporal_responses = []
    valid_labels = []
    
    print(f"Analyzing temporal stability with {len(train_values)} training points...")
    
    for time_offset, label in time_intervals:
        start_idx = time_offset
        end_idx = start_idx + window_size
        
        segment = train_values[start_idx:end_idx]
        segment_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        
        segment_tensor = torch.tensor(segment_norm.reshape(1, 1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            segment_scatter = scattering(segment_tensor)
        
        response_energy = np.abs(segment_scatter.cpu().numpy().flatten())**2
        temporal_responses.append(response_energy)
        valid_labels.append(label)
    
    temporal_array = np.array(temporal_responses)
    log_temporal_array = np.log(temporal_array[:, :100] + 1e-10)
    
    im = plt.imshow(log_temporal_array, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(im, label='Log Response Energy')
    plt.title('WST Temporal Stability Analysis', fontsize=14)
    plt.xlabel('Coefficient Index (first 100)', fontsize=12)
    plt.ylabel('Time Offset from start of the dataset', fontsize=12)
    plt.yticks(np.arange(len(valid_labels)), valid_labels, fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig('scattering_information_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Information compression analysis
    # that is: how many coefficients are needed compared to original signal length
    
    compression_ratio = window_size / len(avg_magnitudes)
    
    effective_compression_90 = window_size / idx_90
    effective_compression_99 = window_size / idx_99
    
    compression_results = {
        'total_coefficients': len(avg_magnitudes),
        'window_size': window_size,
        'raw_compression_ratio': compression_ratio,
        'effective_compression_90': effective_compression_90,
        'effective_compression_99': effective_compression_99,
        'coefficients_90_percent': idx_90,
        'coefficients_99_percent': idx_99,
        'percent_coefficients_90': idx_90 / len(avg_magnitudes) * 100,
        'percent_coefficients_99': idx_99 / len(avg_magnitudes) * 100
    }
    
    print("\n~~~ Scattering info analysis ~~~~")
    print(f"Original signal dimension: {window_size}")
    print(f"Number of scattering coefficients: {len(avg_magnitudes)}")
    print(f"Raw compression ratio: {compression_ratio:.2f}x")
    print(f"90% energy preserved with {idx_90} coefficients ({idx_90/len(avg_magnitudes):.1%})")
    print(f"99% energy preserved with {idx_99} coefficients ({idx_99/len(avg_magnitudes):.1%})")
    print(f"Effective compression ratio (90% energy): {effective_compression_90:.2f}x")
    print(f"Effective compression ratio (99% energy): {effective_compression_99:.2f}x")
    
    return {
        'average_magnitudes': avg_magnitudes,
        'cumulative_energy': cumulative_energy,
        'compression_results': compression_results,
        'temporal_responses': temporal_responses if 'temporal_responses' in locals() else [],
        'temporal_labels': valid_labels if 'valid_labels' in locals() else [],
        'idx_90': idx_90,
        'idx_99': idx_99
    }

def extract_scattering_filters(scattering_transform, scale, signal_length=None):
    """
    Simplified version: Extract filter tensors from Kymatio scattering transform, & graph
    """
    
    if signal_length is None:
        signal_length = scattering_transform.T
    
    J = scattering_transform.J
    Q = scattering_transform.Q
    T = signal_length
    
    if isinstance(Q, tuple):
        Q1, Q2 = Q[0], Q[1] if len(Q) > 1 else 1
    else:
        Q1 = Q
        Q2 = 1
    
    print(f"Filter Analysis: J={J}, Q=({Q1},{Q2}), T={T}, Scale={scale}")
    
    # The loop below just essentially converts any tensors created by the Wavelet Scattering Transform Kymatio package
    # It does this for tensors of a certain size- the window size (T) multiplied by the scaling parameter,
    # As Kymation outputs tensors of different sizes for different j scales
    
    real_filters = []
    tensor_names = []
    
    for attr_name in dir(scattering_transform):
        if attr_name.startswith('tensor') and attr_name[6:].isdigit():
            try:
                tensor = getattr(scattering_transform, attr_name)
                
                if hasattr(tensor, 'shape') and hasattr(tensor, 'detach'):
                    tensor_np = tensor.detach().cpu().numpy()
                    
                    if tensor_np.ndim > 1:
                        tensor_flat = tensor_np.flatten()
                    else:
                        tensor_flat = tensor_np
                    
                    if len(tensor_flat) == T * scale:
                        real_filters.append(tensor_flat)
                        tensor_names.append(attr_name)
                        
            except Exception:
                continue
    
    if len(real_filters) == 0:
        print("no filter tensors found.")
        return None
    
    print(f"Extracted {len(real_filters)} filter tensors")
    
    # Determining filter length for consistent plotting
    # Creating a frequency axis using the fast fourier transform (FFT) based on ze actual filter length

    actual_filter_length = len(real_filters[0]) if real_filters else T

    freqs = np.fft.fftfreq(actual_filter_length, d=1.0)
    freqs_pos = freqs[:actual_filter_length//2]
    
    # Looping through the filters to extract the mathematical properties for each one
    # (so, the center frequency xi, the bandwidth sigma and the scale, j)
    # And then storing these in a list
    
    psi1_f = scattering_transform.psi1_f
    filter_metadata = []
    
    for i, filter_dict in enumerate(psi1_f):
        if isinstance(filter_dict, dict):
            xi = filter_dict.get('xi', 0.0)
            sigma = filter_dict.get('sigma', 0.0)
            j = filter_dict.get('j', 0)
            
            filter_metadata.append({
                'index': i,
                'xi': xi,
                'sigma': sigma, 
                'j': j,
                'center_freq': xi
            })
    
    # And here come the graphs.
    
    plt.figure(figsize=(16, 12))
    colors = plt.cm.viridis(np.linspace(0, 1, len(real_filters)))
    
    matched_metadata = []
    for i in range(len(real_filters)):
        if i < len(filter_metadata):
            matched_metadata.append(filter_metadata[i])
        else:
            matched_metadata.append({'index': i, 'xi': 0.0, 'sigma': 0.0, 'j': 0})
    
    # 1) Filter bank visualisation- frequencey responses for eahc filter
    plt.subplot(2, 2, 1)
    center_frequencies = []
    
    for i, (filter_data, meta, color) in enumerate(zip(real_filters, matched_metadata, colors)):
        filter_magnitude = np.abs(filter_data)
        filter_pos = filter_magnitude[:len(freqs_pos)]
        
        if meta['xi'] > 0:
            label = f"ψ[{i}] (j={meta['j']}, ξ={meta['xi']:.3f})"
            center_frequencies.append(meta['xi'])
        else:
            if np.max(filter_pos) > 0:
                center_idx = np.argmax(filter_pos)
                center_freq = freqs_pos[center_idx]
                center_frequencies.append(center_freq)
                label = f"ψ[{i}] (f₀={center_freq:.3f})"
            else:
                center_frequencies.append(0.0)
                label = f"ψ[{i}]"
        
        plt.plot(freqs_pos, filter_pos, color=color, alpha=0.8, linewidth=1.5, 
                label=label if i < 8 else "")
    
    plt.title(f'Filter Bank - {len(real_filters)} Filters (Tensor Length={int(scale*T)})', fontsize=14)
    plt.xlabel('Frequency')
    plt.ylabel('|Filter(ω)|')
    plt.grid(True, alpha=0.3)
    if len(real_filters) <= 8:
        plt.legend(fontsize=8)
    plt.xlim(0, 0.5)
    
    # 2) Visualising each wavelet in the time domain 
    # (les code below first converts it fro the the frequency domain using the inverse fast fourier transform)

    plt.subplot(2, 2, 2)
    time_axis = np.arange(actual_filter_length)
    
    for i, (filter_data, color) in enumerate(zip(real_filters[:6], colors)):
        filter_time = np.fft.ifft(filter_data)
        filter_real = np.real(filter_time)
        filter_centered = np.fft.ifftshift(filter_real)
        
        min_length = min(len(time_axis), len(filter_centered))
        time_axis_plot = time_axis[:min_length]
        filter_centered_plot = filter_centered[:min_length]
        
        plt.plot(time_axis_plot, filter_centered_plot, color=color, alpha=0.8, 
                linewidth=1.5, label=f'ψ[{i}] (j={matched_metadata[i]["j"]})')
    
    plt.title(f'Wavelets - Time Domain (Tensor Length={int(scale * T)})', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('ψ(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3) FRequency converage... hopefully fairly self explanatory...
    # Coverage energy is based on the square of the filter magnitudes, per wavelet theory
    plt.subplot(2, 2, 3)
    
    total_coverage = np.zeros(len(freqs_pos))
    for filter_data in real_filters:
        filter_magnitude = np.abs(filter_data)
        filter_pos = filter_magnitude[:len(freqs_pos)]
        total_coverage += filter_pos**2
    
    plt.plot(freqs_pos, total_coverage, 'r-', linewidth=2, label='Total Coverage')
    plt.fill_between(freqs_pos, total_coverage, alpha=0.3, color='red')
    plt.xlabel('Frequency')
    plt.ylabel('Coverage Energy')
    plt.title(f'Frequency Coverage (Scale={int(scale * T)})')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.5)
    
    # 4) Frequency vs Bandwidth scatter plot
    # Bandwidth is calculated as the difference between the frequencies at half maximum

    plt.subplot(2, 2, 4)
    
    bandwidths = []
    for i, filter_data in enumerate(real_filters):
        filter_magnitude = np.abs(filter_data)
        filter_pos = filter_magnitude[:len(freqs_pos)]
        
        max_val = np.max(filter_pos)
        if max_val > 0:
            half_max_indices = np.where(filter_pos >= max_val/2)[0]
            if len(half_max_indices) > 1:
                bandwidth = freqs_pos[half_max_indices[-1]] - freqs_pos[half_max_indices[0]]
            else:
                bandwidth = matched_metadata[i]['sigma'] if matched_metadata[i]['sigma'] > 0 else 0.01
        else:
            bandwidth = 0.01
        bandwidths.append(bandwidth)
    
    plt.scatter(center_frequencies, bandwidths, c=[meta['j'] for meta in matched_metadata], 
               cmap='viridis', s=80, alpha=0.8)
    plt.colorbar(label='Scale j')
    plt.xlabel('Center Frequency')
    plt.ylabel('Bandwidth')
    plt.title(f'Frequency vs Bandwidth by Scale (Tensor length={int(scale * T)})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Printing off the results of what filters were actually extracted
    # in the first part of the function for this tensor length
    
    scales = [meta['j'] for meta in matched_metadata]
    
    print(f"\nFilter Summary (Scale={scale}):")
    print(f"{'Index':<5} {'Scale j':<7} {'Center f':<10} {'Bandwidth':<10}")
    print("-" * 40)
    
    for i, (meta, actual_freq, bandwidth) in enumerate(zip(matched_metadata, center_frequencies, bandwidths)):
        print(f"{i:<5} {meta['j']:<7} {actual_freq:<10.4f} {bandwidth:<10.4f}")
    
    print(f"\nScale Distribution:")
    for scale_j in sorted(set(scales)):
        scale_filters = [i for i, s in enumerate(scales) if s == scale_j]
        scale_freq_range = [center_frequencies[i] for i in scale_filters]
        if scale_freq_range:
            print(f"  Scale j={scale_j}: {len(scale_filters)} filters, freq range: {min(scale_freq_range):.4f} - {max(scale_freq_range):.4f}")
    
    return {
        'real_filters': real_filters,
        'tensor_names': tensor_names,
        'filter_metadata': matched_metadata,
        'center_frequencies': center_frequencies,
        'bandwidths': bandwidths,
        'scales': scales,
        'total_extracted': len(real_filters),
        'scale_parameter': scale,
        'actual_filter_length': actual_filter_length
    }