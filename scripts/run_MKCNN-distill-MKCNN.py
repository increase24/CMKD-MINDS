import os
config_file = "./configs/USEMG_cmkd.yaml"
model_us = 'MKCNN'
model_emg = 'MKCNN'
alpha = 0.8
T = 30
os.system('python ./tools/train_cmkd.py --config {} --model_us {} --model_emg {} --alpha {} --T {} '.format(
    config_file, model_us, model_emg, alpha, T))