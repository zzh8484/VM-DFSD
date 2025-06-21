import torch
import torch.nn as nn
import numpy as np
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.myutils import *
from datasets.dataset import *
from sklearn.metrics import confusion_matrix
def prepare_dataloader_val(dataset=None,img_size=256):
    val_dataset =  other_datasets(
        path_Data='data/US30K/'+dataset+'/',
        train=False,
        dataset=dataset,
        image_size=img_size
    )
    return   DataLoader(val_dataset,
                        batch_size=1, 
                        shuffle=False,
                        num_workers=0, 
                        drop_last=True,
                        pin_memory=True,
                        )




def test(name=None, model=None, dataset=None,results_dir=None,img_size=256):
    name = dataset + "_" + name
    log_dir = os.path.join(results_dir, name, "log")
    pth_dir = os.path.join(results_dir, name, "pth")
    os.makedirs(log_dir, exist_ok=True)
    performance_logger = create_logger(
        logger_name="performance",
        filename=os.path.join(log_dir, "performance.log"),  
        level=logging.INFO,
        filemode='a',
        add_console=True
    )
    
    testloader = prepare_dataloader_val(dataset=dataset,img_size=img_size)
    
    model.eval()
    model.cuda()
    
    best_dice = 0.0
    best_checkpoint_path = None
    
    for filename in os.listdir(pth_dir):
        if not filename.endswith('.pth'):
            continue
        checkpoint_path = os.path.join(pth_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        preds = []
        gts = []
        
        with torch.no_grad():
            for batch in tqdm(testloader, desc=f"Testing {filename}"):
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

        print(f"\nProcessing: {filename}")
        logging.info(f"Evaluating checkpoint: {filename}")
        result_str = (
            f"mIoU: {miou*100:.2f}%, Dice (F1): {dice*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%,\n"
            f"F{beta}-Score: {f_beta*100:.2f}%, MAE: {mae*100:.2f}%\n"
            "----------------------------------------"
        )
        print(result_str)
        performance_logger.info(result_str)
        
        if dice > best_dice:
            best_dice = dice
            best_checkpoint_path = checkpoint_path
            best_metrics = {
                'miou': miou,
                'dice': dice,
                'precision': precision,
                'recall': recall,
                'f_beta': f_beta,
                'mae': mae
            }
    if best_checkpoint_path:
        import shutil
        good_model_path = os.path.join(pth_dir, "best_dice.pth")
        shutil.copyfile(best_checkpoint_path, good_model_path)
        best_result_str = (
            f"Best Model Saved: best_dice.pth\n"
            f"mIoU: {best_metrics['miou']*100:.2f}%, Dice (F1): {best_metrics['dice']*100:.2f}%, "
            f"Precision: {best_metrics['precision']*100:.2f}%, Recall: {best_metrics['recall']*100:.2f}%,\n"
            f"F{beta}-Score: {best_metrics['f_beta']*100:.2f}%, MAE: {best_metrics['mae']*100:.2f}%\n"
            "----------------------------------------"
        )
        print(best_result_str)
        performance_logger.info(best_result_str)
if __name__ == "__main__":
    from build_model import build_model
    name="base384"
    model=build_model(model=name,num_classes=1)
    test(model=model, name=name,dataset= "ThyroidNodule-DDTI",results_dir="results/other")
#     "ThyroidNodule-DDTI"
#     "ThyroidNodule-TG3K"
#     "ThyroidNodule-TN3K"
#     "Echocardiography-HMCQU"
#     "Breast-BUSI"
#     "Breast-UDIAT"