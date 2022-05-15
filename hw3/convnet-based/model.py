import torch
import torch.nn as nn

'''
    Example model construction in pytorch
'''
class WDSRblock_typeA(nn.Module):
    def __init__(self, nFeat, ExpandRatio=2):
        super(WDSRblock_typeA, self).__init__()
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat*ExpandRatio, 3, padding=1, bias=True))
        modules.append(nn.ReLU(True))
        modules.append(nn.Conv2d(nFeat*ExpandRatio, nFeat, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        out += x
        return out

class WDSRblock_typeB(nn.Module):
    def __init__(self, nFeat, ExpandRatio=2, bias=True):
        super(WDSRblock_typeB, self).__init__()
        #===== write your model definition here =====#
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat*ExpandRatio, 1, bias=True))
        modules.append(nn.ReLU(True))
        modules.append(nn.Conv2d(nFeat*ExpandRatio, round(nFeat*0.8), 1, bias=True))
        modules.append(nn.Conv2d(round(nFeat*0.8), nFeat, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)
    
    def forward(self, x):
        #===== write your dataflow here =====#
        out = self.body(x)
        out += x
        return out

class upsampler(nn.Module):
    def __init__(self, nFeat, scale=2):
        super(upsampler, self).__init__()
        #===== write your model definition here =====#
        self.layer = nn.Sequential(
            nn.Conv2d(nFeat, (scale**2)*nFeat, 3, padding=1, bias=True), 
            nn.PixelShuffle(2), 
            nn.ReLU(True),
        )
 
    def forward(self, x):
        #===== write your dataflow here =====#
        out = self.layer(x)
        return out

class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, ExpandRatio=4, nResBlock=8, imgChannel=3):
        super(ZebraSRNet, self).__init__()
        #===== write your model definition here using 'WDSRblock_typeB' and 'upsampler' as the building blocks =====#
        self.conv1 = nn.Conv2d(imgChannel, nFeat, 3, padding=1, bias=True)
        self.block = nn.Sequential(
            *nn.ModuleList([WDSRblock_typeB(nFeat, ExpandRatio) for i in range(nResBlock)]))

        self.upsampler = nn.Sequential(
            *nn.ModuleList([upsampler(nFeat, 2) for i in range(2)]))

        self.header = nn.Conv2d(nFeat, imgChannel, 3, padding=1, bias=True)

    def forward(self, x):
        #===== write your dataflow here =====#
        x = self.conv1(x)

        x1 = self.block(x)
        x1 = x1 + x

        x2 = self.upsampler(x1)
        out = self.header(x2)
        return out