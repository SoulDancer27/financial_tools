import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader

""" Data Processing """
class CurrencyDataModule(L.LightningDataModule):
  def __init__(self, data_path, seq_length = 1, columns = None, batch_size = 32, train_length=0.7):
    super().__init__()
    data = pd.read_csv(data_path)
    self.data = data[columns] if columns else data
    self.seq_length = seq_length
    self.train_length = train_length
    self.batch_size = batch_size

  def setup(self, stage):
    training_length = math.floor(len(self.data) * self.train_length)
    train_df, test_df = self.data.iloc[:training_length], self.data.iloc[training_length:]

    if stage == "fit":
      self.train = CurrencyDataset(train_df, self.seq_length)
      self.test = CurrencyDataset(test_df, self.seq_length)

    if stage == "test":
       self.test = CurrencyDataset(test_df, self.seq_length)

    if stage == "predict":
       self.predict = CurrencyDataset(test_df, self.seq_length)

  """ Data Loaders"""
  def train_dataloader(self):
     return DataLoader(self.train, batch_size=self.batch_size)
  
  def test_dataloader(self):
     return DataLoader(self.test, batch_size = self.batch_size)
  
  def predict_dataloader(self):
     return DataLoader(self.test, batch_size=self.batch_size)
  
""" Dataset Logic"""
class CurrencyDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
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