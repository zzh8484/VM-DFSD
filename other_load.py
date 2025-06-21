import argparse
from other_mytest import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', default='/root/haiguang/zzh/dfsd/pth/busi.pth',type=str)
    parser.add_argument('--dataset', default="Breast-BUSI",type=str)
    args = parser.parse_args()
#     "ThyroidNodule-DDTI"
#     "ThyroidNodule-TG3K"
#     "ThyroidNodule-TN3K"
#   "Echocardiography-HMCQU"
#     "Breast-BUSI"
#     "Breast-UDIAT"
    from build_model import build_model
    model=build_model(model='base',num_classes=1,scan_mode=3,k_group=8)

    testloader = prepare_dataloader_val(dataset=args.dataset,img_size=256)
    model.eval()
    checkpoint = torch.load(
        args.pth, 
        map_location=torch.device('cpu')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    preds = []
    gts = []
    with torch.no_grad():
        for batch in testloader:
            img, msk = batch
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            if type(out) is tuple:
                out = out[0]
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds >= 0.5, 1, 0)
    y_true = np.where(gts > 0.5, 1, 0)

    confusion = confusion_matrix(y_true.flatten(), y_pre.flatten())
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    miou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    beta = 1
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) != 0 else 0
    mae = np.mean(np.abs(y_true - y_pre))
    print(   f"mIoU: {miou*100:.2f}%, Dice (F1): {dice*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%,\n"
            f"F{beta}-Score: {f_beta*100:.2f}%, MAE: {mae*100:.2f}%")
    print(f"{args.dataset} has been done")
    print("-"*50)