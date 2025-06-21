import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from functools import partial
from .mamba.vmamba import SS2D
import pywt
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath
import warnings
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WT(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=True, wt_levels=1, wt_type='db1',):
        super(WT, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.align_low = ConvBNReLU(in_channels, in_channels, kernel_size=1)
        self.align_high = ConvBNReLU(in_channels*3 , in_channels, kernel_size=1)
    def forward(self, x):



        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            low = curr_x[:, :, 0, :, :]
            high=curr_x[:, :, 1:4, :, :].reshape(curr_shape[0],curr_shape[1]*3,curr_shape[2]//2,curr_shape[3]//2)
            low = self.align_low(low)     # [B, out_channels, H/2, W/2]
            high = self.align_high(high)  # [B, out_channels, H/2, W/2]

        return low,high

class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Conv2d(channels , channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dec, x):

        # fused =torch.cat((x,dec),dim=1)
        fused=x*dec
        gate = self.sigmoid(self.gate_conv(fused))  # [B, C, H, W]
        out = gate * dec + (1 - gate) * x
        return out
class SimpleQKVGlobalModeling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q(x).flatten(2).transpose(1, 2)  
        k = self.k(x).flatten(2)  
        v = self.v(x).flatten(2).transpose(1, 2) 

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1) 
        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return out



class FSFusionModule(nn.Module):
    def __init__(self, in_channels,scan_mode=3,k_group=8,drop=0):
        super().__init__()
        self.scan_mode=scan_mode
        self.k_group=k_group
        self.drop=drop
        self.local_conv = DW(in_channels=in_channels, out_channels=in_channels, kernel_size=11)
        self.gate1 = GatedFusion(in_channels)    
        self.gate2 = GatedFusion(in_channels)   
        self.wavelet = WT(in_channels=in_channels, out_channels=in_channels)
        self.drop_path = DropPath(self.drop) 
        self.global_atten =SS2D(d_model=in_channels, d_state=1,
        ssm_ratio=2, initialize="v2", forward_type="de_noz", channel_first=True, k_group=self.k_group,scan_mode=self.scan_mode,act_layer=nn.ReLU6)
        # self.global_atten = SimpleQKVGlobalModeling(in_channels)


    def forward(self, feah, feal):
        short_high = feah
        short_low = feal


        feah = self.local_conv(feah)    
        feal = self.global_atten(feal)

        wt=  feal+feah
        low, high = self.wavelet(wt)
        high_upsampled = F.interpolate(high, size=feah.shape[2:], mode='bilinear')
        low_upsampled = F.interpolate(low, size=feah.shape[2:], mode='bilinear')


        feah = self.gate1(high_upsampled, feah)
        feal = self.gate2(low_upsampled, feal)


        feah = self.drop_path(feah)+short_high
        feal = self.drop_path(feal)+short_low

        return feah ,feal 




class DW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DW, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )



    def forward(self, x):
        x = self.up_dwc(x)
        x = self.pwc(x)
        return x




class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7, activation='relu6'):
        super(CBAM, self).__init__()
        
        self.channel_att = ChannelAttention(in_channels, ratio, activation)
        
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16, activation='relu6'):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(1, in_channels // ratio)  
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU6(inplace=True) if activation == 'relu6' else nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x  

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7, 11), 'kernel size must be 3, 7, or 11'
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att_map = torch.cat([avg_out, max_out], dim=1)
        att_map = self.conv(att_map)
        return self.sigmoid(att_map) * x  # 应用注意力权重到输入
    
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = nn.Conv2d(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = DW(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=True):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=True,activation='relu6'):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=True):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )

class DFSD(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64],num_classes=9,scan_mode=3,k_group=8,drop=0):
        super(DFSD, self).__init__()
        self.channel=96
        self.scan_mode=scan_mode
        self.k_group=k_group
        self.drop=drop
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4_1= ConvBNReLU(channels[0], self.channel, kernel_size=1, stride=1)
        self.up3_1= ConvBNReLU(channels[1], self.channel, kernel_size=1, stride=1)
        self.up2_1= ConvBNReLU(channels[2], self.channel, kernel_size=1, stride=1)
        self.fsfm=FSFusionModule(self.channel,scan_mode=self.scan_mode,k_group=self.k_group,drop=self.drop)
        self.cbam=CBAM(self.channel)
        self.wf = WF(in_channels=channels[3],decode_channels=channels[3])
        self.down1=ConvBNReLU(self.channel*3,self.channel, kernel_size=1, stride=1)
        self.down2=ConvBNReLU(self.channel*2,channels[3], kernel_size=1, stride=1)

        self.out_head0 =DW(channels[3], num_classes, kernel_size=1)

    def forward(self,features):

        d4=features[0]
        d3=features[1]
        d2=features[2]
        d1=features[3]
        d4_1=self.up4_1(self.upsample(self.upsample(self.upsample(d4))))
        d3_1=self.up3_1(self.upsample(self.upsample(d3)))
        d2_1=self.up2_1(self.upsample(d2))
        out= torch.cat([d4_1, d3_1, d2_1], dim=1)
        out=self.down1(out)



        out1,out5=self.fsfm(out,out)
        out= self.down2(torch.cat([out1, out5], dim=1))
        d1=self.cbam(d1)
        out= self.wf(out,d1)
        out= self.out_head0(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear')
        return out


if __name__ == '__main__':
    from thop import profile
    skip = []
    skip.append(torch.randn(1, 768, 7, 7).cuda())
    skip.append(torch.randn(1, 384, 14, 14).cuda())
    skip.append(torch.randn(1, 192, 28,28).cuda())
    skip.append(torch.randn(1, 96, 56, 56).cuda())
    model = DFSD(channels= [768, 384, 192, 96]).cuda()
    print(model(skip).shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of parameters (in thousands): {total_params}")