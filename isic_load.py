import argparse
from isic_mytest import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', default='/root/haiguang/zzh/dfsd/pth/isic17.pth',type=str)
    parser.add_argument('--dataset', default='isic2017',type=str)
    args = parser.parse_args()

    from build_model import build_model
    model=build_model(model='base',num_classes=1,scan_mode=3,k_group=8)

    testloader = prepare_dataloader_val(dataset=args.dataset)
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
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
    
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds > 0.5, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    print(f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, acc: {accuracy}, sen: {sensitivity}, spec: {specificity}')
    print(f"{args.dataset} has been done")
    print("-"*50)