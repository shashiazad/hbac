import timm
import torch
import torch.nn as nn
import torchaudio


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

        n, c, h, w = image.size()
        image = image[:, :, :int(20 / 100 * h + 2), :]

        image = torch.clip(image, min=0, max=10000) / 1000

        image = torch.reshape(image, shape=[n, 2, -1, w])

        x1 = image[:, 0:1, ...]
        x2 = image[:, 1:2, ...]

        image = torch.cat([x1, x2], dim=-1)

        image = torch.cat([image, image, image], dim=1)

        return image


class NetOneImage(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()

        self.model = timm.create_model('efficientnet_b5', pretrained=True, in_chans=3)

        self.fc = nn.Linear(2048, 6, bias=True)
        # self.wave_encoder = ResNet1D(inp_ch=42, block=ResNetBlock1D, layers=[2, 2, 4, 2], num_classes=1)

        # weight_init(self.fc)
        self.dropout = nn.Dropout(0.5)

        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # do preprocess
        bs = x.size(0)

        x = self.preprocess(x)

        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc(x)
        x = torch.softmax(x, -1)

        ans = x

        return ans