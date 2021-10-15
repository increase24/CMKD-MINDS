from . import EMGDataloader
from . import USDataloader

def get_dataloader_class(modality):
    return {'EMG': EMGDataloader, 'US': USDataloader}[modality]
