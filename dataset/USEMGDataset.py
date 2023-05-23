from torch.utils.data import Dataset
import numpy as np

class USEMGDataset(Dataset):
    def __init__(self, samples_emg, samples_us, labels):
        self.samples_emg = samples_emg
        self.samples_us = samples_us
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_emg = self.samples_emg[idx]
        sample_us = self.samples_us[idx]
        label = self.labels[idx][0]
        return sample_emg, sample_us, label