import torch
import matplotlib.pyplot as plt
import os

def plot_single_result(
        time_series, 
        next_ground_truth, 
        next_prediction, 
        time_series_names, 
        prefix, 
        save_dir,
        figsize = (10, 3),
        scaler = None
    ):
    '''
    args:
        time_series: (lookback_len, num_variates)
        next_ground_truth: dict (
            k: tensor of shape (k, num_variates) for k in pred_length
        )
        next_prediction: dict (
            k: tensor of shape (k, num_variates) for k in pred_length
        )
        time_series_names: (num_variates,)
        prefix: str
        save_dir: str
    return:
        save results in save_dir
    '''
    pred_lengths = next_ground_truth.keys()
    num_variates = time_series.shape[1]

    series_ground_truth = {
        k: torch.cat((time_series, v)) for k, v in next_ground_truth.items()
    }
    series_prediction = {
        k: torch.cat((time_series, v)) for k, v in next_prediction.items()
    }
  
    if scaler:
        series_ground_truth = {
            k: scaler.inverse_transform(v) for k, v in series_ground_truth.items()
        }
        series_prediction = {
            k: scaler.inverse_transform(v) for k, v in series_prediction.items()
        }

    for k in pred_lengths:
        for i in range(num_variates):
            figure = plt.figure(figsize = figsize)
            fig_title = f'prediction on "{time_series_names[i]}" of {k} next values'
            plt.plot(series_prediction[k][:,i], color = 'red', label = 'prediction')
            plt.plot(series_ground_truth[k][:,i], color = 'green', label = 'ground truth')
            plt.legend()
            plt.title(fig_title)
            save_to = os.path.join(save_dir, prefix + '_' + time_series_names[i] + f'_{k}.png')
            
            plt.savefig(save_to)
            plt.close()
    
def plot_results(
        time_series, 
        next_ground_truths, 
        next_predictions, 
        time_series_names, 
        save_dir,
        figsize = (10,3),
        scaler = None
    ):
    '''
    args:
        time_series: (batch_size, lookback_len, num_variates)
        next_ground_truths: dict (
            k: tensor of shape (batch_size, k, num_variates) for k in pred_length
        )
        next_predictions: dict (
            k: tensor of shape (batch_size, k, num_variates) for k in pred_length
        )
        time_series_names: (num_variates,)
        save_dir: str
    return:
        save results in save_dir
    '''
    os.makedirs(save_dir, exist_ok = True)
    batch_size = time_series.shape[0]
    for i in range(batch_size):
        prefix = f'prediction_{i+1}'
        save_to = os.path.join(save_dir, prefix)
        os.makedirs(save_to, exist_ok = True)
        single_next_ground_truth = {
            k: v[i] for k, v in next_ground_truths.items()
        }
        single_next_prediction = {
            k: v[i] for k, v in next_predictions.items()
        }
        plot_single_result(
            time_series=time_series[i],
            next_ground_truth=single_next_ground_truth,
            next_prediction=single_next_prediction,
            time_series_names=time_series_names,
            prefix = prefix,
            save_dir = save_to,
            figsize = figsize,
            scaler = scaler
        )