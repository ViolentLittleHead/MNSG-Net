import torch
from torch import nn
from CSM import CSM
from torchsummary import summary
from mamba_ssm import Mamba
import torchvision.models as models
from torchprofile import profile_macs

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward(self, x):
        out = self.forward_patch_token(x)
        return out

class DoubleResConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleResConv, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.In = nn.InstanceNorm2d(out_ch)
        self.LR = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, input):
        y = self.In(self.conv1(input))
        y = self.LR(y)
        res = self.In(self.conv2(y)) + self.conv1x1(input)
        return self.LR(res)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, decoder):
        x = self.up(decoder)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class MNSGNet(nn.Module):
    def __init__(self,in_ch,out_ch,img_size=224):
        super(MNSGNet, self).__init__()

        resnet = models.resnet34(pretrained=True)

        self.conv1 = DoubleConv(in_ch, 64)
        self.Maxpool = nn.MaxPool2d(2)
        # =====================================================
        # ResNet34
        # =====================================================
        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4
        # =====================================================
        # MambaLayer
        # =====================================================
        self.mamba = MambaLayer(512)

        self.up6 = Up(512, 256)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = Up(256, 128)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = Up(128, 64)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = Up(64, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # =====================================================
        # CSM Block
        # =====================================================
        self.cATMambaBlock = CSM(img_size,mamba_dim=128 * 4)

    def forward(self, x):
        c1 = self.conv1(x)

        p1 = self.Maxpool(c1)
        c2 = self.Conv2(p1)

        # p2 = self.pool2(c2)
        c3 = self.Conv3(c2)

        # p3 = self.pool3(c3)
        c4 = self.Conv4(c3)

        # p4 = self.pool4(c4)
        c5 = self.Conv5(c4)

        encoder_out = self.mamba(c5) + c5

        c1,c2,c3,c4 = self.cATMambaBlock(c1,c2,c3,c4)

        up_6 = self.up6(encoder_out)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

if __name__=='__main__':
    h, w = 256, 256
    model = MNSGNet(1, 1,img_size=h)
    model = model.to('cuda')
    # summary(model,(1,h,w))
    x1 = torch.rand((1, 1, 256, 256)).to('cuda')
    out = model(x1)
    print(out.shape)

    # 计算 GFLOPs
    # dummy_input = torch.randn(1, 1, h, w).to('cuda')
    # macs = profile_macs(model, dummy_input)
    # gflops = macs / 1e9
    # print(f"GFLOPs: {gflops:.3f}")

