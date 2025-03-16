import torch

from sklearn.preprocessing import MinMaxScaler
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
