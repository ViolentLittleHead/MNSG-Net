import numpy as np
import torch
from torch import nn

from mamba_ssm import Mamba
from torch.nn import Conv2d,Dropout
from torch.nn.modules.utils import _pair

class Mlp(nn.Module):
    """
    mlp_channel 隐藏层
    """
    def __init__(self, in_channel, mlp_channel, dropout_rate=0.1):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

"""
    input (B, n_patch, mambaChannel)
    output (B, decoderChannel, h , w)
"""
class DATMambaLayer(nn.Module):
    def __init__(self, dim, n_patch, skip_connection_nums=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.n_patch = n_patch * 4
        self.cnorm = nn.LayerNorm(dim)
        self.snorm = nn.LayerNorm(self.n_patch)
        self.channel_mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.space_mamba = Mamba(
            d_model=self.n_patch,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.ffn_norm1 = nn.LayerNorm(dim // 4,eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(dim // 4,eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(dim // 4,eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(dim // 4,eps=1e-6)
        self.ffn1 = Mlp(dim // 4, (dim // 4) * 2)
        self.ffn2 = Mlp(dim // 4, (dim // 4) * 2)
        self.ffn3 = Mlp(dim // 4, (dim // 4) * 2)
        self.ffn4 = Mlp(dim // 4, (dim // 4) * 2)

        self.reconstruct = []

        self.reconstruct.append(Reconstruct(dim // skip_connection_nums, 64 , (16 , 16)))
        factor = 1
        for _ in range(skip_connection_nums - 1):
            self.reconstruct.append(Reconstruct(dim // skip_connection_nums, 64 * factor,(8 // factor,8 // factor)))
            factor *= 2
        self.reconstructs = nn.Sequential(*list([m for m in self.reconstruct]))



    def forward(self, x1, x2, x3, x4):
        org1 = x1
        org2 = x2
        org3 = x3
        org4 = x4
        c1 = torch.cat([x1, x2, x3, x4], dim=2)
        m1 = self.channel_mamba(self.cnorm(c1))

        # 使用torch.chunk函数按通道维度（dim=2）将张量分割成4个部分
        cx1,cx2,cx3,cx4 = torch.chunk(m1, chunks=4, dim=2)

        c2 = torch.cat([cx1, cx2, cx3, cx4], dim=1).transpose(-1, -2)

        m2 = self.space_mamba(self.snorm(c2))
        m2 = m2.transpose(-1, -2)
        # 使用torch.chunk函数按通道维度（dim=2）将张量分割成4个部分
        sx1, sx2, sx3, sx4 = torch.chunk(m2, chunks=4, dim=1)
        # 至此，通道注意力和空间注意力都完成
        sx1 = org1 + sx1
        sx2 = org2 + sx2
        sx3 = org3 + sx3
        sx4 = org4 + sx4

        org1 = sx1
        org2 = sx2
        org3 = sx3
        org4 = sx4
        x1 = self.ffn_norm1(sx1)
        x2 = self.ffn_norm2(sx2)
        x3 = self.ffn_norm3(sx3)
        x4 = self.ffn_norm4(sx4)
        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)
        x4 = self.ffn4(x4)
        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3
        x4 = x4 + org4

        split_tensors = x1,x2,x3,x4
        i = 0
        res = []
        for t in split_tensors:
            res.append(self.reconstructs[i](t))
            i += 1
        return (*res,)

class Reconstruct(nn.Module):
    """
    reshape from (B, n_patch, hidden) to (B, hidden, h, w)
    out_channels = [64, 128, 256, 512]
    """
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size = 1):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
            # out = self.up(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Spatial_Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,mamba_inchannel, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=mamba_inchannel // 4,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, mamba_inchannel // 4))

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, mamba_inchannel // 4)
        embeddings = x + self.position_embeddings
        return embeddings


class CSM(nn.Module):
    def __init__(self, img_size, mamba_dim,channel_num = [64, 64, 128, 256], patchSize = [16, 8, 4, 2],skip_connection_nums=4):
        super().__init__()
        self.patchSize_1 = patchSize[0] # 16
        self.patchSize_2 = patchSize[1] # 8
        self.patchSize_3 = patchSize[2] # 4
        self.patchSize_4 = patchSize[3] # 2
        self.embeddings_1 = Spatial_Embeddings(mamba_dim,self.patchSize_1, img_size=img_size,    in_channels=channel_num[0]) # 64
        self.embeddings_2 = Spatial_Embeddings(mamba_dim,self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1]) # 128
        self.embeddings_3 = Spatial_Embeddings(mamba_dim,self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2]) # 256
        self.embeddings_4 = Spatial_Embeddings(mamba_dim,self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3]) # 512
        n_patch = (img_size // self.patchSize_1) * (img_size // self.patchSize_1)
        self.datmamba = DATMambaLayer(mamba_dim,n_patch,skip_connection_nums)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self,en1,en2,en3,en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        o1, o2, o3, o4 = self.datmamba(emb1,emb2,emb3,emb4)

        """
            torch.Size([B, 64, 224, 224])
            torch.Size([B, 128, 112, 112])
            torch.Size([B, 256, 56, 56])
            torch.Size([B, 512, 28, 28])
        """
        return o1, o2, o3, o4


if __name__=='__main__':
    # patch_sizes = [16, 8, 4, 2]
    mamba_dim = 128 * 4
    # channel_num = [64, 128, 256, 512]

    # reconstruct = Reconstruct(32,64,1,(16,16)).to('cuda')
    # res = reconstruct(x1)
    # print(res.shape)

    datm = CSM(224,mamba_dim)
    x1 = torch.rand((1, 64, 224, 224))
    x2 = torch.rand((1, 64, 112, 112))
    x3 = torch.rand((1, 128, 56, 56))
    x4 = torch.rand((1, 256, 28, 28))
    emb1, emb2, emb3, emb4 = datm(x1,x2,x3,x4)
    print(emb1.shape)
    print(emb2.shape)
    print(emb3.shape)
    print(emb4.shape)



