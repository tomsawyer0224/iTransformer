import torch
import lightning as L
import pandas as pd
import yaml
from model.iTransformer import iTransformer
import data
import utils

batch_size = 3
num_variates = 2
time_series_names = [f"price {i}" for i in range(num_variates)]
pred_length = (12,)
lookback_len = 96

time_series = torch.randn(size=(batch_size, lookback_len, num_variates))
next_ground_truths = {
    k: torch.rand(size=(batch_size, k, num_variates)) for k in pred_length
}
next_predictions = {
    k: torch.rand(size=(batch_size, k, num_variates)) for k in pred_length
}

utils.plot_results(
    time_series=time_series,
    next_ground_truths=next_ground_truths,
    next_predictions=next_predictions,
    time_series_names=time_series_names,
    save_dir="./test_results",
)


"""
data_file = './datasets/AAPL.csv'
lookback_len = 3
pred_length = (4,5)

config_file = './config.yaml'
with open(config_file, 'r') as file:
    configs = yaml.safe_load(file)
pred_length = configs['general_config']['pred_length']
print(type(pred_length))
print(pred_length)



ts_datamodule = data.TimeSeriesDataModule(
    data_file = data_file,
    lookback_len = lookback_len,
    pred_length = pred_length
)
train_dataloader = ts_datamodule.train_dataloader()
x, y = next(iter(train_dataloader))
#print(f'x: {x.shape}\n', x)
#print(f'len(y): {len(y)}, shape: {[yi.shape for yi in y]}\n', y)
#print("-"*30)

test_dataloader = ts_datamodule.test_dataloader()
x, y = next(iter(test_dataloader))
print(f'x: {x.shape}\n', x)
print(f'len(y): {len(y)}, shape: {[yi.shape for yi in y]}\n', y)

raw_df = pd.read_csv(data_file)._get_numeric_data()
train_df = raw_df[:200]
train_dataset = data.TimeSeriesDataset(
    df = train_df,
    lookback_len = lookback_len,
    pred_length = pred_length,
    scaler = None
)
scaler = train_dataset.scaler
test_df = raw_df[200:300]
test_dataset = data.TimeSeriesDataset(
    df = train_df,
    lookback_len = lookback_len,
    pred_length = pred_length,
    scaler = scaler
)
for i, (x, y) in enumerate(train_dataset):
    print(x)
    print("-"*30)
    print(y)
    print("#"*30)
    if i==2: break
print("*"*40)
for i, (x, y) in enumerate(test_dataset):
    print(x)
    print("-"*30)
    print(y)
    print("#"*30)
    if i==2: break


model = iTransformer(
    num_variates = 6,
    lookback_len = lookback_len, #96,               # or the lookback length in the paper
    dim = 256,                       # model dimensions
    depth = 6,                       # depth
    heads = 8,                       # attention heads
    dim_head = 64,                   # head dimension
    pred_length = pred_length, #(12, 24, 36, 48),  # can be one prediction, or many
    num_tokens_per_variate = 1       # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
)

time_series = x #torch.randn(2, lookback_len, 6)  # (batch, lookback len, variates)

preds = model(time_series, y)
#print([preds[k].dtype for k in preds.keys()])
print(preds)
"""
