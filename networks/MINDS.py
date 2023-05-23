import torch
import torch.nn as nn
from torch.nn.modules import adaptive, module


class DiverseFocusModule(nn.Module):
    def __init__(self, c_in, n_filters, ks_base, stride) -> None:
        super(DiverseFocusModule, self).__init__()
        # extract spatial-temporal feature
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=n_filters, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, groups=n_filters, \
                kernel_size=(1, ks_base), stride=(1, stride), padding=(0,ks_base//2))
        )
        
        # only extract spatial feature
        self.path2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1, stride), padding=(0,1)),
            nn.Conv2d(in_channels=c_in, out_channels=n_filters, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        )

        # extract spatial-temporal feature
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=n_filters, kernel_size=(1,ks_base), stride=(1,stride), padding=(0,ks_base//2)),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        )

        # only extract temporal feature
        self.path4 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(in_channels=c_in, out_channels=n_filters, kernel_size=(1,ks_base), stride=(1,stride), padding=(0, ks_base//2))
        )

    def forward(self, x):
        stream1 = torch.cat([self.path1(x), self.path2(x)], dim=1)
        stream2 = torch.cat([self.path3(x), self.path4(x)], dim=1)
        out = torch.cat([stream1, stream2], dim=1)
        return out

class DiverseFocusBlock(nn.Module):
    def __init__(self, c_in, n_filter, ks_base, stride, shortcut) -> None:
        super(DiverseFocusBlock, self).__init__()
        self.path1 = nn.Sequential(
            DiverseFocusModule(c_in, n_filter, ks_base, stride)
        )
        self.shortcut = shortcut
        if(shortcut):
            self.path2 = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=n_filter*4, kernel_size=(1,3), stride=(1,stride), padding=(0,1)),
                nn.BatchNorm2d(num_features=n_filter*4)
            )

    def forward(self, x):
        y1 = self.path1(x)
        if not(self.shortcut):
            return y1
        else:
            y2 = self.path2(x)
            out = y1 + y2
            return out


class MINDS(nn.Module):
    def __init__(self, cfg) -> None:
        super(MINDS, self).__init__()
        in_channels = cfg.in_channels
        n_filters = cfg.n_filters
        strides = cfg.strides
        shortcuts = cfg.shortcuts
        ks_base = cfg.ks_base
        adaptivePoolSize = cfg.adaptivePoolSize
        lc1 = cfg.lc1
        lc2 = cfg.lc2
        num_classes = cfg.num_classes

        layers = []
        for i in range(len(n_filters)):
            layers.append(DiverseFocusBlock(c_in=in_channels, n_filter=n_filters[i], \
                 ks_base=ks_base, stride=strides[i], shortcut=shortcuts[i]))
            layers.append(nn.ReLU(inplace=True))
            in_channels = n_filters[i]*4
        self.cnn_encoder = nn.Sequential(*layers)
        self.net = nn.Sequential(
            self.cnn_encoder,
            nn.AdaptiveAvgPool2d(output_size=(4, adaptivePoolSize)),
            nn.Conv2d(in_channels=n_filters[-1]*4, out_channels=lc1, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=lc1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=lc1, out_channels=lc2, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=lc2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=lc2, out_channels=num_classes, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=num_classes),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

    def forward(self, x):
        #x = x.squeeze(1) # b,1,c,w->b,c,w
        out = self.net(x)
        out = out.squeeze(-1).squeeze(-1)
        return out


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    from thop import profile
    cfg = CN.load_cfg(open('./configs/MINDS.yaml'))
    # test on sEMG modality
    model = MINDS(cfg['EMG'])
    model.to("cuda")
    input = torch.randn((1,1,4,256), dtype=torch.float32)
    input = input.to("cuda")
    output = model(input)
    print(output.shape)
    macs, params = profile(model, inputs=(input, ))
    print('macs:', macs, 'params:', params)
    # test on AUS modality
    model = MINDS(cfg['US'])
    model.to("cuda")
    input = torch.randn((1,1,4,1024), dtype=torch.float32)
    input = input.to("cuda")
    output = model(input)
    print(output.shape)
    macs, params = profile(model, inputs=(input, ))
    print('macs:', macs, 'params:', params)