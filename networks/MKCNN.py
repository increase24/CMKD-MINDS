import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

class MKBlock(nn.Module):
    def __init__(self, blockCfg) -> None:
        super(MKBlock, self).__init__()
        self.streams = nn.ModuleList()
        for idx in range(blockCfg.num_stream):
            streamCfg = blockCfg['stream' + str(idx+1)]
            self.streams.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=blockCfg.in_channels, out_channels= blockCfg.num_filters1,\
                        kernel_size=(3, streamCfg.conv_ksize1), stride=(1,1), padding=(1, streamCfg.conv_ksize1//2)),
                    nn.ELU(inplace=True),
                    nn.BatchNorm2d(blockCfg.num_filters1),
                    nn.MaxPool2d(kernel_size=(1, streamCfg.maxpool_ksize1)),
                    nn.Dropout(p=blockCfg.dropout_rate),
                    nn.Conv2d(in_channels=blockCfg.num_filters1, out_channels= blockCfg.num_filters2,\
                        kernel_size=(3, streamCfg.conv_ksize2), stride=(1,1), padding=(1, streamCfg.conv_ksize2//2)),
                    nn.ELU(inplace=True),
                    nn.BatchNorm2d(blockCfg.num_filters2),
                    nn.MaxPool2d(kernel_size=(streamCfg.maxpool_ksize2, streamCfg.maxpool_ksize2)),
                    nn.Dropout(p=blockCfg.dropout_rate)
                )
            )
    def forward(self, x):
        out_streams = []
        for idx in range(len(self.streams)):
            out_streams.append(self.streams[idx](x))
        out = torch.cat(out_streams, dim=1)
        return out
        


class MKCNN(nn.Module):
    def __init__(self, cfg):
        super(MKCNN, self).__init__()
        blockCfg = cfg.Block
        self.part1 = MKBlock(blockCfg)
        out_dim_block = blockCfg.num_stream * blockCfg.num_filters2
        self.part2 = nn.Sequential(
            nn.Conv2d(out_dim_block, cfg.num_filters_reduce, kernel_size=1, stride=1),
            nn.ELU(inplace=True)
        )
        self.part3 = nn.Sequential(
            nn.Conv2d(cfg.num_filters_reduce, cfg.num_filters_reduce, groups=cfg.num_filters_reduce,\
                kernel_size=(3,3), stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(cfg.num_filters_reduce),
            nn.MaxPool2d(kernel_size=(2, cfg.maxpool_ksize1_part3)),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Conv2d(cfg.num_filters_reduce, cfg.num_filters_reduce, groups=cfg.num_filters_reduce,\
                kernel_size=(1,3), stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, cfg.maxpool_ksize1_part3)),
            nn.BatchNorm2d(cfg.num_filters_reduce),
            nn.Dropout(p=cfg.dropout_rate),
        )
        self.part4 = nn.Sequential(
            nn.Linear(cfg.featDim_concat, cfg.fc1),
            nn.ELU(inplace=True),
            nn.Linear(cfg.fc1, cfg.fc2),
            nn.ELU(inplace=True),
            nn.Linear(cfg.fc2, cfg.num_classes)
        )

    def forward(self, x):
        output = self.part1(x)
        output = self.part2(output)
        output = self.part3(output)
        output = output.view(output.shape[0], -1)
        output = self.part4(output)
        return output
        

if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    config = CN.load_cfg(open('./configs/MKCNN.yaml'))
    model = MKCNN(config['EMG'])
    model.eval()
    input = torch.randn((8,1,4,256))
    output = model(input)
    print(output.shape)