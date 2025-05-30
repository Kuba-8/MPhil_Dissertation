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

def visualize_scattering_information(scattering, data, scattering_params, window_size, sample_indices=None, num_samples=5):
    """
    A bit of an EDA (exploratory data ananlysis) function for graphing the data,
    alongside the information extracted by the wavelet scattering transform
    (e.g. energy content of the coefficients)
    
    Visualize the information content extracted by wavelet scattering transforms.
    
    """
    if isinstance(data, pd.DataFrame):
        data_values = data.values
        time_index = data.index
    else:
        data_values = data
        time_index = range(len(data.flatten()))
    
    # Does not want to use my GPU for some reason, so set a backup
    try:
        device = next(scattering.parameters()).device
        print(f"Using device: {device}")
    except (StopIteration, AttributeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using default device: {device}")
    
    # Some simple, raw data plots to begin with

    plt.figure(figsize=(22, 20))

    # Two years of raw tiem series data, for two random years (e.g. 2016-18)

    plt.subplot(5, 2, 1)
    data_flat = data_values.flatten()
    
    start_2016 = pd.to_datetime('2016-01-01')
    end_2018 = pd.to_datetime('2018-01-01')
    mask_2016_2018 = (data.index >= start_2016) & (data.index < end_2018)
    subset_data = data[mask_2016_2018]
    plt.plot(subset_data.index, subset_data.values.flatten())
    plt.title(f'Energy Data 2016-2018 ({len(subset_data)} points)', fontsize=14)
    plt.xlabel('Date/Time')
    plt.ylabel('Energy Load (kWh)')
    plt.grid(True, alpha=0.3)
    
    # All (training) data in single plot... should I just use the whole dataset?

    plt.subplot(5, 2, 2)
    plt.plot(time_index, data_flat, alpha=0.7, linewidth=0.5)
    plt.title(f'Complete Dataset - All {len(data_flat)} Points (~20 Years)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Energy Load (kWh)')
    plt.grid(True, alpha=0.3)
    
    # ACF plot, using first 5000 points and seasonally adjusted series (with CI's)
    # (plotted it before, and ACF just went up and down like the signal, which is unhelpful)
    plt.subplot(5, 2, 3)
    acf_data = data_flat[:min(5000, len(data_flat))]
    
    acf_data_seasonal = acf_data[24:] - acf_data[:-24]
    acf_series = acf_data_seasonal
    series_description = "Seasonally Adjusted"

    acf_values = acf(acf_series, nlags=min(168, len(acf_series)//4), fft=True)  # Up to 1 week lag
    
    plt.plot(range(len(acf_values)), acf_values, 'b-', alpha=0.8, linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    n = len(acf_series)
    confidence_interval = 1.96 / np.sqrt(n)
    plt.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.5, label='95% Confidence')
    plt.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f'ACF - {series_description} Series', fontsize=14)
    plt.xlabel('Lag (hours)')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Superimposed 3 months from the middle of the period, in order to attempt 
    # to show long-memory characteristic of the data

    plt.subplot(5, 2, 4)
    month_length = 24 * 30
    
    start_idx = len(data_flat) // 3
    colors_months = plt.cm.Set1(np.linspace(0, 1, 3))
    
    for i in range(3):
        month_start = start_idx + i * month_length
        month_end = month_start + month_length
        month_data = data_flat[month_start:month_end]
        plt.plot(range(month_length), month_data, alpha=0.8, 
                color=colors_months[i], label=f'Month {i+1}', linewidth=1.5)
    
    plt.title('Superimposed 3 Consecutive Months', fontsize=14)
    plt.xlabel('Hours in Month')
    plt.ylabel('Energy Load (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Again, trying to show long dependence, this time with daily averaged-metered hourly load 
    # across 3 consecutive years chosen at random (2012, 2013, 2014)

    plt.subplot(5, 2, 5)
    years_to_plot = [2012, 2013, 2014]
    colors_years = plt.cm.Set1(np.linspace(0, 1, 3))
    
    for i, year in enumerate(years_to_plot):
        year_start = pd.to_datetime(f'{year}-01-01')
        year_end = pd.to_datetime(f'{year+1}-01-01')
        year_mask = (data.index >= year_start) & (data.index < year_end)
        year_data = data[year_mask].values.flatten()
        daily_avg = [np.mean(year_data[j:j+24]) for j in range(0, len(year_data), 24)]
        plt.plot(range(len(daily_avg)), daily_avg, alpha=0.8, 
                color=colors_years[i], label=f'Year {year}', linewidth=1.5)
    
    plt.title(f'Superimposed Years: 2012, 2013, 2014 (Daily Averages)', fontsize=14)
    plt.xlabel('Days in Year')
    plt.ylabel('Average Daily Energy Load (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Testing out the scattering transform on the first few "windows" of the training data
    if sample_indices is None:
        max_idx = len(data_values) - window_size
        sample_indices = np.linspace(0, max_idx, num_samples).astype(int)
    
    print(f"Analyzing {len(sample_indices)} samples with window size {window_size}")
    
    samples = []
    scattering_outputs = []
    
    plt.subplot(5, 2, 6)
    
    for i, idx in enumerate(sample_indices):
        sample = data_values[idx:idx + window_size].flatten()
        samples.append(sample)
        
        sample_norm = (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
        
        sample_tensor = torch.tensor(sample_norm.reshape(1, 1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            scatter_output = scattering(sample_tensor)
            scattering_outputs.append(scatter_output.cpu().numpy())
        
        if i < 5:
            plt.plot(sample_norm, alpha=0.7, label=f'Sample {i+1}')
    
    plt.title('Sample Windows for Scattering Analysis')
    plt.xlabel('Time Points')
    plt.ylabel('Normalized Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Showing daily seasonality for the first month (4 weeks) of data

    plt.subplot(5, 2, 7)
    sample_size = min(24*7*4, len(data_flat))
    seasonal_data = data_flat[:sample_size]
    
    days_to_show = min(7, len(seasonal_data) // 24)
    colors_days = plt.cm.viridis(np.linspace(0, 1, days_to_show))
    
    for day in range(days_to_show):
        day_start = day * 24
        day_end = day_start + 24
        day_data = seasonal_data[day_start:day_end]
        plt.plot(range(24), day_data, alpha=0.8, 
                color=colors_days[day], label=f'Day {day+1}')
    
    plt.title('Daily Patterns (First Week)', fontsize=12)
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Load (kWh)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Energy load distribution histogram
    plt.subplot(5, 2, 8)
    plt.hist(data_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Energy Load Distribution', fontsize=14)
    plt.xlabel('Energy Load (kWh)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    mean_val = np.mean(data_flat)
    median_val = np.median(data_flat)
    std_val = np.std(data_flat)
    
    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
    plt.legend()
    
    # Now, starting a separate figure for the wavelet scattering analysis

    plt.figure(figsize=(16, 12))
    
    # Taking one of the couple of scattering tranforms computed,
    # and averaging the magnitudes of the generated coefficients

    scattering_array = np.array([s.flatten() for s in scattering_outputs])
    avg_magnitudes = np.mean(np.abs(scattering_array), axis=0)
    
    # Just plotting the avg magnitude of each coefficient
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(avg_magnitudes)), avg_magnitudes, 'o-', alpha=0.7)
    plt.yscale('log')
    plt.title('Scattering Coefficient Magnitude (Log Scale)', fontsize=14)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Average Magnitude')
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
    plt.xlabel('Number of Coefficients')
    plt.ylabel('Cumulative Energy Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Data frequency analysis - showing scattering response to different time periods/periodicities
    # NEEDS FIXING, COME BACK TO IT

    plt.subplot(2, 2, 3)
    
    time_periods = ['Start of Dataset (t=0)', 'One Day Later (t+24h)', 'One Week Later (t+168h)', 'Two Weeks Later (t+336h)', 'End of Dataset (t=final)', 'Middle of Dataset (t=50%)']
    period_responses = []
    
    daily_pattern = 24
    weekly_pattern = 24 * 7
    
    # Looking at 6 different segments of the data, with various offsets from each other
    segments = [
        data_values[0:window_size].flatten(),
        data_values[daily_pattern:daily_pattern+window_size].flatten(),
        data_values[weekly_pattern:weekly_pattern+window_size].flatten(),
        data_values[2*weekly_pattern:2*weekly_pattern+window_size].flatten(),
        data_values[-window_size:].flatten(),
        data_values[len(data_values)//2:len(data_values)//2+window_size].flatten()
    ]
    
    # Finding the response energies, which are just squares of the magnitudes of the scattering coefficients
    # See Mallat (2012) or Bolliet (2024) within my paper

    for i, segment in enumerate(segments):
        segment_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        
        segment_tensor = torch.tensor(segment_norm.reshape(1, 1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            segment_scatter = scattering(segment_tensor)
        
        response_energy = np.abs(segment_scatter.cpu().numpy().flatten())**2
        period_responses.append(response_energy)
    
    # Plotting logged versions as a heatmap (logged because it makes it look nicer)
    resp_array = np.array(period_responses)
    log_resp_array = np.log(resp_array + 1e-10)
    
    plt.imshow(log_resp_array[:, :100], aspect='auto', interpolation='nearest')
    plt.colorbar(label='Log Response Energy')
    plt.title('Scattering Response to Real Data Patterns', fontsize=14)
    plt.xlabel('Coefficient Index (first 100)')
    plt.ylabel('Data Segment Type')
    actual_labels = time_periods[:len(period_responses)]
    plt.yticks(np.arange(len(actual_labels)), actual_labels)
    
    # Temporal analysis - scattering response to patterns at different times
    # (trying to show stability of scatteirng coefficients over time/translation invariance!)

    plt.subplot(2, 2, 4)
    
    positions = [0, len(data_values)//4, len(data_values)//2, 3*len(data_values)//4, len(data_values)-window_size]
    position_labels = ['Start', 'Quarter', 'Middle', '3/4', 'End']
    
    position_responses = []
    
    for pos, label in zip(positions, position_labels):
        real_segment = data_values[pos:pos + window_size].flatten()
        
        segment_norm = (real_segment - np.mean(real_segment)) / (np.std(real_segment) + 1e-8)
        
        segment_tensor = torch.tensor(segment_norm.reshape(1, 1, -1), dtype=torch.float32).to(device)
        with torch.no_grad():
            segment_scatter = scattering(segment_tensor)
        
        response = np.abs(segment_scatter.cpu().numpy().flatten())
        position_responses.append(response)
    
    pos_array = np.array(position_responses)
    plt.imshow(pos_array[:, :100], aspect='auto', interpolation='nearest')
    plt.colorbar(label='Response Magnitude')
    plt.title('Scattering Consistency Across Time Periods', fontsize=14)
    plt.xlabel('Coefficient Index (first 100)')
    plt.ylabel('Time Period in Dataset')
    plt.yticks(np.arange(len(position_labels)), position_labels)
    
    plt.tight_layout()
    plt.savefig('scattering_information_analysis.png', dpi=300)
    plt.show()
    
    # Data time scales visualization

    J = scattering_params['J']
    Q = scattering_params['Q']
    
    plt.figure(figsize=(14, 10))
    
    scale_segments = []
    scale_names = []
    
    hourly_scale = 24
    weekly_scale = 24 * 7 
    monthly_scale = 24 * 30
    
    time_scales = [
        (hourly_scale, "Daily Pattern (24h)"),
        (weekly_scale, "Weekly Pattern (168h)"),
        (monthly_scale, "Monthly Pattern (720h)"),
    ]
    
    # sourced from different parts of the dataset

    for i, (scale, name) in enumerate(time_scales):
        start_idx = i * scale
        segment = data_values[start_idx:start_idx + window_size].flatten()
        scale_segments.append(segment)
        scale_names.append(name)
    
    # Adding additional segments from different seasons/periods

    additional_segments = [
        ("Early Period", data_values[:window_size].flatten()),
        ("Mid Period", data_values[len(data_values)//2:len(data_values)//2+window_size].flatten()),
        ("Late Period", data_values[-window_size:].flatten()),
    ]
    
    for name, segment in additional_segments:
        scale_segments.append(segment)
        scale_names.append(name)
    
    # Plot the real data segments showing different time scales,
    # for a data length of a week

    num_plots = len(scale_segments)
    
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i+1)
        segment = scale_segments[i]
        name = scale_names[i]
        
        display_length = min(len(segment), 168)
        plt.plot(range(display_length), segment[:display_length])
            
        plt.title(f"{name} - Real Energy Data")
        plt.ylabel("Energy (kWh)")
        plt.grid(True, alpha=0.3)
        
        if i == num_plots - 1:
            plt.xlabel("Time (hours)")
    
    plt.tight_layout()
    plt.savefig('time_scales_visualization.png', dpi=300)
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
    
    print("\n===== Scattering Information Analysis =====")
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
        'position_responses': position_responses,
        'idx_90': idx_90,
        'idx_99': idx_99
    }
