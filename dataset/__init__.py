from . import EMGDataloader
from . import USDataloader
from . import USEMGDataloader

def get_dataloader_class(modality):
    return {'EMG': EMGDataloader, 'US': USDataloader, 'USEMG': USEMGDataloader}[modality]
