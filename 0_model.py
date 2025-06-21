import torch
from build_model import *
import numpy as np
if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 224,224).cuda()
    model=build_model(model='base',num_classes=1).cuda()
    # =====================================================================
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    flops = FlopCountAnalysis(model, input_tensor)
    print("FLOPs: ", flops.total() / 1e9)
    print(parameter_count_table(model)) 
    # =====================================================================
    from thop import profile
    flops, params = profile(model, inputs=(input_tensor, ))
    print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) 
    #=====================================================================
    print(np.unique(model(input_tensor).cpu().detach().numpy()))

        
