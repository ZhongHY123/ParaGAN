import math

import torch.nn as nn

def upsample(x, size):
    x_up = nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up

class ESPCN(nn.Module):
    def __init__(self, inchannel,out_channels=1, scale_factor=4):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.05, inplace=True),


            nn.Conv2d(inchannel, inchannel, kernel_size=3),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.05, inplace=True),
        )
        self.body_part = nn.Sequential(
            nn.Conv2d(inchannel, out_channels * (scale_factor ** 2), kernel_size=3),
            nn.PixelShuffle(scale_factor),

        )

        self.last_part = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.05,inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.05, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(inchannel),
            nn.LeakyReLU(0.05, inplace=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x_1 = self.first_part(x)
        x_2 = self.body_part(x_1)
        x_3 = upsample(x_2,[x_2.shape[2]+6,x_2.shape[3]+6])
        x_4 = self.last_part(x_3)
        return x_2
