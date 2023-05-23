import os
import argparse
import glob
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
from munch import Munch
os.sys.path.append('.')
from dataset import get_dataloader_class
from networks import get_network
from utils.trainer import Trainer_kd
from utils.saver import save_checkpoint, save_result
from utils.loss import loss_cls_kd


str2bool = lambda x: (str(x).lower() == 'true')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training of CMKD')
    parser.add_argument('--model_emg', help='model of sEMG stream')
    parser.add_argument('--model_us', help='model of AUS stream')
    parser.add_argument('--config', help='json config file path')
    parser.add_argument('--alpha', default = 0.5, type = float, help='Weights of KL-loss')
    parser.add_argument('--T', default = 5, type = float, help='Softening factor')
    args = parser.parse_args()
    return args

def main():
    train_start_time = time.time()
    curr_time = time.strftime('%m-%d-%H-%M', time.localtime())
    args = parse_args()
    with open(args.config) as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        print('Successfully loading the config file....')
        opts = [
            'ModelConfig.model_us', args.model_us, 
            'ModelConfig.model_emg', args.model_emg,
            ]
        cfg.merge_from_list(opts)
        print(cfg)
    device = torch.device('cuda:0')
    ModelConfig = cfg.ModelConfig
    paths_EMGData = sorted(glob.glob('./data/EMG/*.txt'))
    paths_USData = sorted(glob.glob('./data/US/*.txt'))
    class_dataloader = get_dataloader_class(ModelConfig.modality)
    OutputConfig = cfg.OutputConfig
    results = np.zeros((len(paths_USData), 4))
    if os.path.exists(os.path.join(OutputConfig.dir_results, ModelConfig.model_us+'-kd-'+ModelConfig.model_emg,\
         ModelConfig.model_us+'-kd-'+ModelConfig.model_emg+ '_EMG_KD_a{:.2f}_T{}.txt'.format(args.alpha, args.T))):
        results = np.loadtxt(os.path.join(OutputConfig.dir_results, ModelConfig.model_us+'-kd-'+ModelConfig.model_emg,\
             ModelConfig.model_us+'-kd-'+ModelConfig.model_emg+ '_EMG_KD_a{:.2f}_T{}.txt'.format(args.alpha, args.T)))
    for idx_subject in range(len(paths_EMGData)):
        for cross_val in range(4): # 4-fold
            print(f"subject: {idx_subject}, cv: {cross_val}")
            # load dataloader
            print("Start loading the dataloader....")
            DataConfig = cfg.DatasetConfig
            train_loader, valid_loader = class_dataloader.get_dataloader(paths_EMGData[idx_subject], 
                paths_USData[idx_subject], cross_val, DataConfig)
            print('Finish loading the dataloader....')
            # load network
            with open(ModelConfig.model_arch[ModelConfig.model_us]) as data_file:
                modelCfg_us = CN.load_cfg(data_file)
            with open(ModelConfig.model_arch[ModelConfig.model_emg]) as data_file:
                modelCfg_emg = CN.load_cfg(data_file)
            model_us = get_network(ModelConfig.model_us, modelCfg_us['US'])
            model_us.to(device)
            checkpoint = torch.load(f'./outputs/weights/{ModelConfig.model_us}/{ModelConfig.model_us}_US_s{idx_subject}_cv{cross_val}.pth.tar')
            model_us.load_state_dict(checkpoint['state_dict'])
            model_emg = get_network(ModelConfig.model_emg, modelCfg_emg['EMG'])
            model_emg.to(device)
            # define criterion, optimizer, scheduler
            OptimizerConfig = cfg.OptimizerConfig
            # criterion = nn.CrossEntropyLoss().to(device)
            params = Munch({'alpha':args.alpha, 'temperature':args.T})
            criterion =  loss_cls_kd(params)
            optimizer = torch.optim.AdamW(model_emg.parameters(), lr=OptimizerConfig.lr)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.5)
            num_epoches = int(OptimizerConfig["epoches"])
            trainer = Trainer_kd(train_loader, valid_loader, model_emg, model_us, device, \
                criterion, optimizer, print_freq=50)
            print(" > Training is getting started...")
            print(" > Training takes {} epochs.".format(num_epoches))
            for epoch in range(num_epoches):
                # train one epoch
                epoch_start_time = time.time()
                train_loss, train_acc = trainer.train_kd_epoch(epoch) 
                valid_loss, valid_acc = trainer.validate_kd(eval_only=False)
                epoch_end_time = time.time()
                print("epoch cost time: %.4f min" %((epoch_end_time - epoch_start_time)/60))
                #scheduler.step()
                print(f'current best accuracy: {trainer.bst_acc}')
                # remember best acc and save checkpoint
                if(trainer.flag_improve):
                    print(f'the best accuracy increases to {trainer.bst_acc}')
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': ModelConfig.model_us+'-kd-'+ModelConfig.model_emg,
                        'state_dict': trainer.model_stu.state_dict(),
                        'best_acc': trainer.bst_acc}, 
                        os.path.join(OutputConfig.dir_weights, ModelConfig.model_us+'-kd-'+ModelConfig.model_emg), 
                        ModelConfig.model_us+'-kd-'+ModelConfig.model_emg+f'_EMG_s{idx_subject}_cv{cross_val}' +'.pth.tar')
            results[idx_subject,cross_val] = trainer.bst_acc
            save_result(results, os.path.join(OutputConfig.dir_results, ModelConfig.model_us+'-kd-'+ModelConfig.model_emg), ModelConfig.model_us+'-kd-'+ModelConfig.model_emg+ 
                '_EMG_KD_a{:.2f}_T{}.txt'.format(args.alpha, args.T))
    print('acc-avg-s:\n', np.mean(results, 1))
    print('acc-avg-total:\n', np.mean(np.mean(results, 1)))
    train_end_time = time.time()
    print("total training time: %.2f min" %((train_end_time - train_start_time)/60))

if __name__ == "__main__":
    main()
    