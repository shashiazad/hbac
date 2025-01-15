import sys

sys.path.append('.')

import timm

import torchaudio
import torch.nn as nn
import torch


class Transform50s(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(n_fft=512, win_length=128, hop_length=50, power=1)

    def forward(self, x):
        image = self.wave_transform(x)
        image = torch.clip(image, min=0, max=10000) / 1000
        n, c, h, w = image.size()
        image = image[:, :, :int(20 / 100 * h + 10), :]
        return image


class Transform10s(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(n_fft=512, win_length=128, hop_length=10, power=1)

    def forward(self, x):
        image = self.wave_transform(x)
        image = torch.clip(image, min=0, max=10000) / 1000
        n, c, h, w = image.size()
        image = image[:, :, :int(20 / 100 * h + 10), :]
        return image


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        model_name = "x3d_l"
        self.net = torch.hub.load('facebookresearch/pytorchvideo',
                                  model_name, pretrained=True)

        self.net.blocks[5].pool.pool = nn.AdaptiveAvgPool3d(1)
        # self.net.blocks[5]=nn.Identity()
        # self.net.avgpool = nn.Identity()
        self.net.blocks[5].dropout = nn.Identity()
        self.net.blocks[5].proj = nn.Identity()
        self.net.blocks[5].activation = nn.Identity()
        self.net.blocks[5].output_pool = nn.Identity()

    def forward(self, x):
        x = self.net(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess50s = Transform50s()
        self.preprocess10s = Transform10s()

        self.model = Model()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.drop = nn.Dropout(0.5)

    def forward(self, eeg):
        # do preprocess
        bs = eeg.size(0)
        eeg_50s = eeg
        eeg_10s = eeg[:, :, 4000:6000]
        x_50 = self.preprocess50s(eeg_50s)
        x_10 = self.preprocess10s(eeg_10s)
        x = torch.cat([x_10, x_50], dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.cat([x, x, x], dim=1)
        x = self.model(x)
        # x = self.pool(x)
        x = x.view(bs, -1)
        x = self.drop(x)
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
