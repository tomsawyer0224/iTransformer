import torch
import yaml
import argparse
import os
from tqdm import tqdm


import lightning as L

from model.iTransformer import iTransformer
import data
import utils


class Training:
    def __init__(self, config_file, extra_args):
        with open(config_file, "r") as file:
            configs = yaml.safe_load(file)
        self.configs = configs
        self.extra_args = extra_args

    def run(self):
        # extra args
        ckpt_path = self.extra_args.ckpt_path
        max_epochs = self.extra_args.max_epochs

        # general config
        general_config = self.configs["general_config"]
        if isinstance(general_config["pred_length"], list):
            general_config["pred_length"] = tuple(general_config["pred_length"])
        # dataset
        dataset_config = self.configs["dataset_config"]
        dataset_config = dataset_config | general_config  # concat 2 dicts
        # model config
        model_config = self.configs["model_config"]
        # training config
        training_config = self.configs["training_config"]
        training_config["max_epochs"] = max_epochs

        # data module
        ts_datamodule = data.TimeSeriesDataModule(**dataset_config)

        # model
        model_config["num_variates"] = ts_datamodule.num_variates
        model_config = model_config | general_config
        ts_model = iTransformer(**model_config)

        # trainer
        trainer = L.Trainer(**training_config)
        trainer.fit(
            model=ts_model, train_dataloaders=ts_datamodule, ckpt_path=ckpt_path
        )

        trainer.test(model=ts_model, dataloaders=ts_datamodule)
        test_dir = os.path.join(training_config["default_root_dir"], "tests")
        os.makedirs(test_dir, exist_ok=True)

        test_dataloader = ts_datamodule.test_dataloader()

        time_series_names = ts_datamodule.time_series_names
        scaler = ts_datamodule.scaler

        for batch_idx, batch in tqdm(
            enumerate(test_dataloader), desc="Generating test results"
        ):
            batch_dir = os.path.join(test_dir, f"batch_{batch_idx+1}")
            os.makedirs(batch_dir, exist_ok=True)

            time_series, next_ground_truths = batch
            with torch.no_grad():
                next_predictions = ts_model(time_series)
            utils.plot_results(
                time_series=time_series,
                next_ground_truths=next_ground_truths,
                next_predictions=next_predictions,
                time_series_names=time_series_names,
                save_dir=batch_dir,
                figsize=(10, 3),
                scaler=scaler,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--ckpt_path", default=None)
    extra_args = parser.parse_args()

    config_file = "./config.yaml"
    training_ts_model = Training(config_file=config_file, extra_args=extra_args)

    # run the training
    training_ts_model.run()
