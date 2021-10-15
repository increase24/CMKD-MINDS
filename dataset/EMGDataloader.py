import pandas as pd
import numpy as np
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.sys.path.append('.')
from dataset.EMGDataset import EMGDataset

def get_dataloader(data_path, group_test, DataConfig):
    df_emgData=pd.read_csv(data_path,sep=',',header = None) #(n_sample,960*4)
    df_emgData=df_emgData.iloc[:,4:8]
    df_emgData.rename(columns={4:0,5:1,6:2,7:3}, inplace=True)
    # normalization
    df_emgData = (df_emgData-df_emgData.mean())/(df_emgData.std())
    _max=max(abs(df_emgData.max(axis=0).max()),abs(df_emgData.min(axis=0).min()))
    df_emgData= df_emgData/_max
    df_emgData=df_emgData.append(pd.DataFrame(np.zeros((156,4))),ignore_index=True)
    data_org = np.zeros((10000,1,4,256))
    for i in range(data_org.shape[0]):
        data_org[i,:,:,:] = df_emgData.iloc[100*i:100*i+256,:].values.transpose().reshape([1,1,4,256])
    #exact 4800 valid samples from raw 10000 samples
    indexs_list=[]
    for i in range(8): # 8 trial
        for j in range(20): # 20 motions
            indexs = [i for i in range(i*1250+j*50+10,i*1250+j*50+40)]
            indexs_list=indexs_list+indexs
    data_org=data_org[indexs_list,:,:,:]
    data_org = data_org.astype('float32')
    labels = np.linspace(0,19,20, endpoint=True, dtype=int)
    labels = np.tile(labels, (30,1))
    labels= labels.reshape(-1,1,order = 'F')
    labels = np.tile(labels, (8,1))
    #split train and test dataset (4-fold)
    group_train=[0,1,2,3]
    group_train.remove(group_test)
    data_train = data_org[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                    group_train[1]*1200:(group_train[1]+1)*1200,
                                    group_train[2]*1200:(group_train[2]+1)*1200],:,:,:]
    data_test = data_org[group_test*1200:(group_test+1)*1200,:,:,:]
    label_train = labels[np.r_[group_train[0]*1200:(group_train[0]+1)*1200,
                                        group_train[1]*1200:(group_train[1]+1)*1200,
                                        group_train[2]*1200:(group_train[2]+1)*1200],:]
    label_test = labels[group_test*1200:(group_test+1)*1200,:]
    train_data = EMGDataset(data_train, label_train)
    valid_data = EMGDataset(data_test, label_test)
    train_loader = DataLoader(train_data, batch_size = DataConfig.batch_size, shuffle = True, num_workers = DataConfig.num_workers, 
        pin_memory = True, drop_last = True)
    valid_loader = DataLoader(valid_data, batch_size = DataConfig.batch_size, shuffle = False, num_workers = DataConfig.num_workers,
        pin_memory = True, drop_last = False)
    return train_loader, valid_loader

if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    paths_EMGData = sorted(glob.glob('./data/EMG/*.txt'))
    idx_subject = 0
    idx_cv = 3
    with open('./configs/USEMG_single.yaml') as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        DataConfig = cfg.DatasetConfig
    train_loader, valid_loader = get_dataloader(paths_EMGData[idx_subject], idx_cv, DataConfig)
    samples, labels = iter(valid_loader).next()
    print(samples.shape, labels.shape)
    fig = plt.figure()
    color_list = [77,175,74]
    color_list = [_color/255.0 for _color in color_list]
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(samples[0][0][0].numpy(), color = color_list)
    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(samples[0][0][1].numpy(), color = color_list)
    ax3 = fig.add_subplot(4,1,3)
    ax3.plot(samples[0][0][2].numpy(), color = color_list)
    ax4 = fig.add_subplot(4,1,4)
    ax4.plot(samples[0][0][3].numpy(), color = color_list)
    #plt.show()
    plt.savefig('./figs/EMGSample.png', dpi=600)

