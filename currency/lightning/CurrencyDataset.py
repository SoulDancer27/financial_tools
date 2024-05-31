import torch
import pandas as pd
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, data, seq_length = 1, columns = None):
        """
        Arguments:
            columns: list of columns from a dataframe
        """
        data = data
        self.data = data[columns] if columns else data
        self.seq_length = seq_length
        self.total_length = len(data) - seq_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        if index < self.total_length:
            sequence = self.data.iloc[index: index + self.seq_length].values
            target = self.data.iloc[index + self.seq_length].values
            return [torch.tensor(sequence, dtype=torch.float32),
                     torch.tensor(target, dtype=torch.float32)]