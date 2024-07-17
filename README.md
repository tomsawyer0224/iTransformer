# This is a personal project, for educational purposes only!
# About this project:
1. iTransformer or Inverted Transformer is a Time Series Forcasting model. The model use a different embedding method. It embeds each series independently to the variate token.
2. This project is a small version converted to the Lightning model.
# How to use:
1. Clone this repo, then cd to iTransformer.
2. Install the requirements: pip install -q -r requirements.txt.
3. Modify the config file (./config.yaml), then run the below command:
```
!python train.py \
  --max_epochs 3 \
  --ckpt_path 'path/to/checkpoint # add this line if resume the training from a checkpoint
```
> See the "./results" directory for more details.
![image](https://github.com/user-attachments/assets/e9532e7d-a6c8-4bb1-ac0a-5c0047f105c6) \
![image](https://github.com/user-attachments/assets/3bfbcea9-f019-4adf-9e58-5b4d81ea02f2) \
# Based on:
https://arxiv.org/abs/2310.06625
https://github.com/thuml/iTransformer
https://github.com/lucidrains/iTransformer/tree/main


