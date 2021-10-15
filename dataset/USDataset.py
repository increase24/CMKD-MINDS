from torch.utils.data import Dataset
import numpy as np

class USDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx][0]
        return sample, label
