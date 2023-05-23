from .EUNet import EUNet
from .MSCNN import MSCNN
from .XceptionTime import XceptionTime
from .MKCNN import MKCNN
from .MINDS import MINDS


def get_network(name, param):
    model = {'EUNet':EUNet, 'MSCNN':MSCNN, 'XceptionTime':XceptionTime, 'MKCNN':MKCNN, 'MINDS':MINDS}[name]
    return model(param)
