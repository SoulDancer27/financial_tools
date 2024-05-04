import torch
import pandas as pd
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, csv_file):
        """
        Arguments:
            csv_file (string): Path to the csv file with data
        """
        data = pd.read_csv(csv_file)
        self.data = data[['open', 'high', 'close', 'vol']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].values)