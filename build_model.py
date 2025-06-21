from lib.networks import VMDFSD

def build_model(model=None,num_classes=None,scan_mode=3,k_group=8,pretain="True",device='cuda:0',pretrained_dir="./pre/vssm_small_0229_ckpt_epoch_222.pth"):
    if 'base' in model:
        model = VMDFSD(encoder="base",num_classes=num_classes,pretain=pretain,scan_mode=scan_mode,k_group=k_group,pretrained_dir=pretrained_dir).to(device)  
    elif "light" in model:
        model = VMDFSD(encoder="light",num_classes=num_classes).to(device)
    return model
if __name__ == '__main__':
    build_model()