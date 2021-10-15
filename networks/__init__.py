from .EUNet import EUNet
from .MSCNN import MSCNN
from .XceptionTime import XceptionTime
from .MKCNN import MKCNN


def get_network(name, param):
    model = {'EUNet':EUNet, 'MSCNN':MSCNN, 'XceptionTime':XceptionTime, 'MKCNN':MKCNN}[name]
    return model(param)
