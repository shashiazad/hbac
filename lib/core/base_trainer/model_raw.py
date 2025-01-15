import timm
import torch
import torch.nn as nn


## inp   1x16x10000

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True, in_chans=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, out_features=6, bias=True)
        self.dropout = nn.Dropout(p=0.5)

    def extract_features(self, x):
        feature1 = self.model.forward_features(x)
        return feature1

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
        x = self.dropout(x)
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
