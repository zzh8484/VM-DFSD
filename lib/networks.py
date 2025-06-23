import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoders import DFSD
from .mamba.vmamba import *

class VMDFSD(nn.Module):
    def __init__(self, num_classes=9, encoder='base', pretain="True", pretrained_dir="./pre/vssm_small_0229_ckpt_epoch_222.pth",scan_mode=3,k_group=8):
        super(VMDFSD, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU6(inplace=True)
        )
        self.scan_mode=scan_mode
        self.k_group=k_group

        if pretain=="True":
            pretrained=pretrained_dir
            print("load pretrained model")
        else:
            pretrained=None
        if not os.path.exists(pretrained_dir):
            raise FileNotFoundError(f"no finding: {pretrained_dir}")
        if encoder=="base":
            self.backbone= Backbone_VSSM(
                depths=[2, 2, 2, 2], dims=96, drop_path_rate=0.3, 
                patch_size=4, in_chans=3, num_classes=1000, 
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
                ssm_init="v0", forward_type="v05_noz", 
                mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                patch_norm=True, norm_layer=("ln2d" if True else "ln"), 
                downsample_version="v3", patchembed_version="v2", 
                use_checkpoint=False, posembed=False, imgsize=224, 
                pretrained=pretrained
                ) 
            channels=[768, 384, 192, 96]

        elif encoder=="light":
            self.backbone= Backbone_VSSM(
                depths=[1, 1, 1, 1], dims=96, drop_path_rate=0.3, 
                patch_size=4, in_chans=3, num_classes=1000, 
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
                ssm_init="v0", forward_type="v05_noz", 
                mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                patch_norm=True, norm_layer=("ln2d" if True else "ln"), 
                downsample_version="v3", patchembed_version="v2", 
                use_checkpoint=False, posembed=False, imgsize=224, 
                pretrained=pretrained
                ) 
            channels=[768, 384, 192, 96]

        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            
        # self.decoder = DFSD(channels=channels,num_classes=num_classes,scan_mode=self.scan_mode,k_group=self.k_group)
        
        self.decoder = DFSD(channels=channels,num_classes=num_classes)
        print('Model %s created, param count: %f' %
                     ('decoder: ', sum([m.numel() for m in self.decoder.parameters()])/1e6))

        
    def forward(self, x, mode='test'):
        
        if x.size()[1] == 1:
            x = self.conv(x)
        
        x1, x2, x3, x4 = self.backbone(x)

        dec_outs = self.decoder([x4,x3, x2, x1])
        
        if self.num_classes == 1: return torch.sigmoid(dec_outs)

        
        return dec_outs
               

        
if __name__ == '__main__':
    model = vmdfsd().cuda()
    input_tensor = torch.randn(1, 1, 224, 224).cuda()


    print(model(input_tensor).shape)

