import argparse
from syn_mytest import *
from build_model import build_model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', default='./pth/syn.pth',type=str)
    parser.add_argument('--only', default='True',type=str)
    args = parser.parse_args()
    model=build_model(model='base',num_classes=9,scan_mode=3,k_group=8)

    testloader = prepare_dataloader_val()
    model.eval()
    checkpoint = torch.load(
        args.pth, 
        map_location=torch.device('cpu')
    )
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    if args.only=='True':
        performance, class_metric= inference(model, testloader,only_dice=True)
    else:
        performance, class_metric= inference(model, testloader,only_dice=False)
    print(f"Dice: {performance[0]*100:.2f}")
    for idx, dice in enumerate(class_metric, 1):
        print(f"Class {idx} Dice: {dice[0]*100:.2f},hd5:  {dice[1]:.2f},jaccard: {dice[2]*100:.2f},asd: {dice[3]:.2f}")
    print(f"syn has been done")
    print("-"*50)

