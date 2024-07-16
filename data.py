from pandas.io import parsers
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightning as L

class TimeSeriesDataset(Dataset):
    def __init__(self, df, lookback_len, pred_length, scaler = None):
        if not isinstance(pred_length, tuple): # if pred_length = int
            pred_length = (pred_length,)
        assert len(df) >= lookback_len + max(pred_length), 'need more data'
        if not scaler:
            scaler = StandardScaler().fit(df)
            #scaler.fit(df)
        self.scaler = scaler
        self.df = self.scaler.transform(df)
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        self.num_variates = df.shape[1]
    def __len__(self):
        return len(self.df) - self.lookback_len - max(self.pred_length)
    def __getitem__(self, idx):
        x = self.df[idx:idx+self.lookback_len]
        y = {
            k: self.df[idx+self.lookback_len:idx+self.lookback_len+k] for k in self.pred_length
        }
        # convert to torch tensor
        '''
        x = torch.tensor(x).float() # float32
        y = tuple([
            torch.tensor(v).float() for v in y.values() # float32
        ])
        '''
        x = torch.tensor(x).float() # float32
        y = {
            k: torch.tensor(v).float() for k, v in y.items() # float32
        }
        return x, y

class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, data_file, lookback_len, pred_length):
        super().__init__()
        if not isinstance(pred_length, tuple): # if pred_length = int
            pred_length = (pred_length,)
        self.raw_df = pd.read_csv(data_file)
        self.lookback_len = lookback_len
        self.pred_length = pred_length
        df = self.raw_df._get_numeric_data()
        self.time_series_names = list(df.columns)
        self.train_dataset = TimeSeriesDataset(
            #df = self.raw_df._get_numeric_data()[:-lookback_len-max(pred_length)-4],
            df = df[:-lookback_len-max(pred_length)-4],
            lookback_len = lookback_len,
            pred_length = pred_length,
            scaler = None
        )
        self.scaler = self.train_dataset.scaler
        self.test_dataset = TimeSeriesDataset(
            #df = self.raw_df._get_numeric_data()[-lookback_len-max(pred_length)-4:],
            df = df[-lookback_len-max(pred_length)-4:],
            lookback_len = lookback_len,
            pred_length = pred_length,
            scaler = self.scaler
        )
        self.num_variates = self.train_dataset.num_variates
    @staticmethod
    def collate_fn(examples):
        '''
        args:
            examples: list of examples (example = (tensor, tuple of tensors))
        return:
            stacked tensor, tuple of stacked tensor
        
        # i.e: batch_size = 4, pred_length = (5,6) -> len = 2
        x_list = [exam[0] for exam in examples] # list of 4 tensors
        y_list = [exam[1] for exam in examples] # list of 4 tuples, each tuple = (2 tensors)
        #convert to torch tensor
        x = torch.stack(x_list)
        y = list(zip(*y_list)) # list of 2 objects
        y = [torch.stack(yi) for yi in y]
        y = tuple(y)
        '''
        x, y = zip(*examples)
        x = torch.stack(x)
        y = {
            k: torch.stack([
                yi[k] for yi in y
            ]) for k in y[0].keys()
        }
        return x, y
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            collate_fn = self.collate_fn
        )
    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = 2,
            shuffle = False,
            num_workers = 2,
            collate_fn = self.collate_fn
        )
        