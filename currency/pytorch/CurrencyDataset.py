import torch
import pandas as pd
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, data, seq_length = 1, columns = None):
        """
        Arguments:
            csv_file (string): Path to the csv file with data
            columns: list of columns from a dataframe
        """
        data = data
        self.data = data[columns] if columns else data
        self.seq_length = seq_length
        self.total_length = len(data) - seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index < self.total_length:
            sequence = self.data.iloc[index: index + self.seq_length].values
            target = self.data.iloc[index + self.seq_length].values
            return {'data': torch.tensor(sequence), 'target': torch.tensor(target, dtype=torch.float32)}