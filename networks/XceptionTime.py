import torch
import torch.nn as nn
from torch.nn.modules import adaptive, module

class MultiStreamSeparableConv(nn.Module):
    def __init__(self, n_filters, ks_base) -> None:
        super(MultiStreamSeparableConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, groups=n_filters, kernel_size=ks_base+1, stride=1, padding=ks_base//2)
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, groups=n_filters, kernel_size=ks_base*2+1, stride=1, padding=ks_base//2*2)
        self.conv3 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, groups=n_filters, kernel_size=ks_base*4+1, stride=1, padding=ks_base//2*4)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        out = torch.cat([y1, y2, y3], dim=1)
        return out

class XceptionTimeModule(nn.Module):
    def __init__(self, c_in, n_filters, ks_base) -> None:
        super(XceptionTimeModule, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=n_filters, kernel_size=1, stride=1),
            MultiStreamSeparableConv(n_filters, ks_base)
        )
        self.path2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=c_in, out_channels=n_filters, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        out = torch.cat([y1, y2], dim=1)
        return out

class XceptionTimeBlock(nn.Module):
    def __init__(self, c_in, n_filter1, n_filter2, ks_base) -> None:
        super(XceptionTimeBlock, self).__init__()
        self.path1 = nn.Sequential(
            XceptionTimeModule(c_in, n_filter1, ks_base),
            XceptionTimeModule(n_filter1*4, n_filter2, ks_base)
        )
        self.path2 = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=n_filter2 * 4, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=n_filter2 * 4)
        )

    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        out = y1 + y2
        return out
    

class XceptionTime(nn.Module):
    def __init__(self, cfg) -> None:
        super(XceptionTime, self).__init__()
        in_channels = cfg.in_channels
        n_filter1_block1 = cfg.n_filter1_block1
        n_filter2_block1 = cfg.n_filter2_block1
        n_filter1_block2 = cfg.n_filter1_block2
        n_filter2_block2 = cfg.n_filter2_block2
        ks_base = cfg.ks_base
        adaptivePoolSize = cfg.adaptivePoolSize
        lc1 = cfg.lc1
        lc2 = cfg.lc2
        num_classes = cfg.num_classes
        self.net = nn.Sequential(
            XceptionTimeBlock(c_in=in_channels, n_filter1=n_filter1_block1, n_filter2=n_filter2_block1, ks_base=ks_base),
            nn.ReLU(inplace=True),
            XceptionTimeBlock(c_in=n_filter2_block1*4, n_filter1=n_filter1_block2, n_filter2=n_filter2_block2, ks_base=ks_base),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=adaptivePoolSize),
            nn.Conv1d(in_channels=n_filter2_block2*4, out_channels=lc1, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=lc1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=lc1, out_channels=lc2, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=lc2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=lc2, out_channels=num_classes, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_features=num_classes),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        x = x.squeeze(1) # b,1,c,w->b,c,w
        out = self.net(x)
        out = out.squeeze(-1)
        return out


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    cfg = CN.load_cfg(open('./configs/XceptionTime.yaml'))
    model = XceptionTime(cfg['EMG'])
    model.to("cuda")
    input = torch.randn((8,1,4,256), dtype=torch.float32)
    input = input.to("cuda")
    output = model(input)
    print(output.shape)
