import math
import sys

sys.path.append('.')

import timm

import torchaudio
import torch.nn as nn
import torch
import random


class Transform(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.wave_transform = torchaudio.transforms.Spectrogram(n_fft=512,
                                                                win_length=128,
                                                                hop_length=50,
                                                                power=1)

    def forward(self, x):
        bs = x.size(0)

        image = self.wave_transform(x)

        image = torch.clip(image, min=0, max=10000) / 1000

        n, c, h, w = image.size()

        ## inference use 0-20hz filter,
        image = image[:, :, :int(20 / 100 * h + 2), :]

        return image


class Modeleeg(nn.Module):
    def __init__(self, ):
        super(Modeleeg, self).__init__()

        self.model = timm.create_model('efficientnet_b5',
                                       pretrained=True,
                                       in_chans=3, )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, out_features=6, bias=True)

    def extract_features(self, x):
        x = self.model.forward_features(x)

        return x

    def forward(self, x):
        bs = x.size(0)
        reshaped_tensor = x.view(bs, 16, 1000, 10)

        reshaped_and_permuted_tensor = reshaped_tensor.permute(0, 1, 3, 2)

        reshaped_and_permuted_tensor = reshaped_and_permuted_tensor.reshape(bs, 16 * 10, 1000)

        x = torch.unsqueeze(reshaped_and_permuted_tensor, dim=1)

        x = torch.cat([x, x, x], dim=1)
        bs = x.size(0)

        x = self.extract_features(x)

        x = self.pool(x)
        x = x.view(bs, -1)

        return x


class Modelspec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        model_name = "x3d_l"
        self.net = torch.hub.load('facebookresearch/pytorchvideo',
                                  model_name, pretrained=True)


        self.net.blocks[5].pool.pool=nn.AdaptiveAvgPool3d(1)
        # self.net.blocks[5]=nn.Identity()
        # self.net.avgpool = nn.Identity()
        self.net.blocks[5].dropout = nn.Identity()
        self.net.blocks[5].proj = nn.Identity()
        self.net.blocks[5].activation = nn.Identity()
        self.net.blocks[5].output_pool = nn.Identity()

        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = torch.cat([x, x, x], dim=1)
        x = self.net(x)
        # do preprocess
        bs = x.size(0)

        x = x.view(bs, -1)

        return x


class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.transform = Transform()
        self.model_wave = Modeleeg()

        self.model_spec = Modelspec()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2048 * 2, 6, bias=True)

        self.droup = nn.Dropout(0.5)

    def forward(self, eeg):
        # do preprocess
        bs = eeg.size(0)

        eeg_spec = self.transform(eeg)

        x = self.model_wave(eeg)
        y = self.model_spec(eeg_spec)
        x = torch.cat([x, y], dim=1)

        x = self.droup(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    from thop import clever_format

    dummy_data = torch.randn(1, 4, 100, 300, device='cpu')
    dummy_waves = torch.randn(1, 16, 10000, device='cpu')
    model = Net()

    # x = model( dummy_waves)
    macs, params = profile(model, inputs=[dummy_waves])
    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)
