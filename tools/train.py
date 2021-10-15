import os
import sys
import argparse
import glob
import shutil
import time
import json
import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
# custom
os.sys.path.append('.')
from dataset import get_dataloader_class
from networks import get_network
from utils.trainer import Trainer
from utils.saver import save_checkpoint, save_result
from utils.initializer import weight_init


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Jester Training using JPEG')
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--modelName', help='name of nn model')
    parser.add_argument('--modality', help='modality to train(EMG, US or USEMG)')
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
            'ModelConfig.model_name', args.modelName, 
            'ModelConfig.modality', args.modality
            ]
        cfg.merge_from_list(opts)
        print(cfg)
    device = torch.device('cuda:0')
    ModelConfig = cfg.ModelConfig
    OutputConfig = cfg.OutputConfig
    paths_data = sorted(glob.glob('./data/'+ModelConfig.modality+'/*.txt'))
    class_dataloader = get_dataloader_class(ModelConfig.modality)
    results = np.zeros((len(paths_data), 4))
    if os.path.exists(os.path.join(OutputConfig.dir_results, ModelConfig.model_name, ModelConfig.model_name+ f'_{ModelConfig.modality}.txt')):
        results = np.loadtxt(os.path.join(OutputConfig.dir_results, ModelConfig.model_name, ModelConfig.model_name+ f'_{ModelConfig.modality}.txt'))
    start_idx_subject, end_idx_subject = (0, 8)
    start_cv, end_cv = (0, 4)
    for idx_subject in range(start_idx_subject, end_idx_subject): #range(len(paths_data)):
        for cross_val in range(start_cv, end_cv): #range(4): # 4-fold
            print(f"subject: {idx_subject}, cv: {cross_val}")
            # load dataloader
            print("Start loading the dataloader....")
            DataConfig = cfg.DatasetConfig
            train_loader, valid_loader = class_dataloader.get_dataloader(paths_data[idx_subject], cross_val, DataConfig)
            print('Finish loading the dataloader....')
            # load network
            with open(ModelConfig.model_arch[ModelConfig.model_name]) as data_file:
                cfg_model = CN.load_cfg(data_file)
                cfg_model = cfg_model[ModelConfig.modality]
            model = get_network(ModelConfig.model_name, cfg_model)
            model.apply(weight_init)
            model.to(device)
            # define criterion, optimizer, scheduler
            OptimizerConfig = cfg.OptimizerConfig
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=OptimizerConfig.lr)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70,90], gamma=0.1)
            # start training
            num_epoches = int(OptimizerConfig.epoches)
            trainer = Trainer(train_loader, valid_loader, model, device, criterion, optimizer, print_freq=100)
            print(" > Training is getting started...")
            print(" > Training takes {} epochs.".format(num_epoches))
            # trainer.reset_optimiser(optimizer)
            for epoch in range(num_epoches):
                # train one epoch
                epoch_start_time = time.time()
                train_loss, train_acc = trainer.train_epoch(epoch) 
                valid_loss, valid_acc = trainer.validate(eval_only=False)
                epoch_end_time = time.time()
                print("epoch cost time: %.4f min" %((epoch_end_time - epoch_start_time)/60))
                #scheduler.step()
                print(f'current best accuracy: {trainer.bst_acc}')
                # remember best acc and save checkpoint
                if(trainer.flag_improve):
                    print(f'the best accuracy increases to {trainer.bst_acc}')
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': ModelConfig['model_name'],
                        'state_dict': trainer.model.state_dict(),
                        'best_acc': trainer.bst_acc}, 
                        os.path.join(OutputConfig.dir_weights, ModelConfig.model_name),
                        ModelConfig.model_name+f'_{ModelConfig.modality}_s{idx_subject}_cv{cross_val}'+'.pth.tar')
                #scheduler.step()
            results[idx_subject,cross_val] = trainer.bst_acc
            save_result(results, os.path.join(OutputConfig.dir_results, ModelConfig.model_name), ModelConfig.model_name+ f'_{ModelConfig.modality}.txt')
    #print('acc s/cv:\n', results)
    print('acc-avg-s:\n', np.mean(results, 1))
    print('acc-avg-total:\n', np.mean(np.mean(results, 1)))
    train_end_time = time.time()
    print("total training time: %.2f min" %((train_end_time - train_start_time)/60))

if __name__ == "__main__":
    main()
    