import pandas as pd
import numpy as np
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.sys.path.append('.')
from dataset.USEMGDataset import USEMGDataset

def get_dataloader(data_path_emg, data_path_us, group_test, DataConfig):

    ''' load emg data set ''' 
    df_emgData=pd.read_csv(data_path_emg, sep=',',header = None) #(n_sample,960*4)
    df_emgData=df_emgData.iloc[:,4:8]
    df_emgData.rename(columns={4:0,5:1,6:2,7:3}, inplace=True)
    # normalization
    df_emgData = (df_emgData-df_emgData.mean())/(df_emgData.std())
    _max=max(abs(df_emgData.max(axis=0).max()),abs(df_emgData.min(axis=0).min()))
    df_emgData= df_emgData/_max
    df_emgData=df_emgData.append(pd.DataFrame(np.zeros((156,4))),ignore_index=True)
    data_org_emg = np.zeros((10000,1,4,256))
    for i in range(data_org_emg.shape[0]):
        data_org_emg[i,:,:,:] = df_emgData.iloc[100*i:100*i+256,:].values.transpose().reshape([1,1,4,256])
    #exact 4800 valid samples from raw 10000 samples
    indexs_list=[]
    for i in range(8): # 8 trial
        for j in range(20): # 20 motions
            indexs = [i for i in range(i*1250+j*50+10,i*1250+j*50+40)]
            indexs_list=indexs_list+indexs
    data_org_emg=data_org_emg[indexs_list,:,:,:]
    data_org_emg = data_org_emg.astype('float32')

    ''' load us data set ''' 
    df_usData=pd.read_csv(data_path_us, sep=',',header = None) #(n_sample,960*4)
    # normalization
    df_usData = (df_usData-df_usData.mean())/(df_usData.std())
    _max=max(abs(df_usData.max(axis=0).max()),abs(df_usData.min(axis=0).min()))
    df_usData= df_usData/_max
    data_org_us = df_usData.values.reshape((4800,1,4,960))
    data_org_us = np.concatenate([data_org_us, np.zeros((4800,1,4,64))], axis = -1)
    data_org_us = data_org_us.astype('float32')

    ''' load labels ''' 
    labels = np.linspace(0,19,20, endpoint=True, dtype=int)
    labels = np.tile(labels, (30,1))
    labels= labels.reshape(-1,1,order = 'F')
    labels = np.tile(labels, (8,1))

    ''' split train and test dataset (4-fold) ''' 
    group_train=[0,1,2,3]
    group_train.remove(group_test)
    data_train_emg = data_org_emg[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                    group_train[1]*1200:(group_train[1]+1)*1200,
                                    group_train[2]*1200:(group_train[2]+1)*1200],:,:,:]
    data_test_emg = data_org_emg[group_test*1200:(group_test+1)*1200,:,:,:]
    data_train_us = data_org_us[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                    group_train[1]*1200:(group_train[1]+1)*1200,
                                    group_train[2]*1200:(group_train[2]+1)*1200],:,:,:]
    data_test_us = data_org_us[group_test*1200:(group_test+1)*1200,:,:,:]
    label_train = labels[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                        group_train[1]*1200:(group_train[1]+1)*1200,
                                        group_train[2]*1200:(group_train[2]+1)*1200],:]
    label_test = labels[group_test*1200:(group_test+1)*1200,:]
    train_data = USEMGDataset(data_train_emg, data_train_us, label_train)
    valid_data = USEMGDataset(data_test_emg, data_test_us, label_test)
    train_loader = DataLoader(train_data, batch_size = DataConfig.batch_size, shuffle = True, num_workers = DataConfig.num_workers,
        pin_memory = True, drop_last = True)
    valid_loader = DataLoader(valid_data, batch_size = DataConfig.batch_size, shuffle = False, num_workers = DataConfig.num_workers,
        pin_memory = True, drop_last = False)
    return train_loader, valid_loader


