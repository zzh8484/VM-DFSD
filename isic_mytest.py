import torch
import torch.nn as nn
import numpy as np
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.myutils import *
from datasets.dataset import NPY_datasets
from sklearn.metrics import confusion_matrix

def prepare_dataloader_val(dataset=None):
    val_dataset = NPY_datasets(
        path_Data='/root/haiguang/zzh/datasets/'+dataset+'/',
        train=False,
        dataset=dataset
    )
    return   DataLoader(val_dataset,
                        batch_size=1, 
                        shuffle=False,
                        num_workers=4, 
                        drop_last=True,
                        pin_memory=True,
                        )




def test(name=None, model=None, dataset=None,results_dir=None):
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
    
    testloader = prepare_dataloader_val(dataset=dataset)
    
    model.eval()
    model.cuda()
    
    best_dice = 0.0
    best_checkpoint_path = None
    best_metrics = None
    

    for filename in os.listdir(pth_dir):
        if not filename.endswith('.pth'):
            continue
        checkpoint = torch.load(os.path.join(pth_dir, filename), map_location='cuda')
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
        
        result_str = (
            f"\nProcessing: {filename} Evaluating checkpoint: {filename}"
            f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, acc: {accuracy}, sen: {sensitivity}, spec: {specificity}\n'
            "----------------------------------------"
        )
        print(result_str)
        performance_logger.info(result_str)
        
        if f1_or_dsc > best_dice:
            best_dice = f1_or_dsc
            best_checkpoint_path = os.path.join(pth_dir, filename)
            best_metrics = {
                'miou': miou,
                'dice': f1_or_dsc,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
    
    if best_checkpoint_path:
        import shutil
        best_model_path = os.path.join(pth_dir, "best_dice.pth")
        shutil.copyfile(best_checkpoint_path, best_model_path)
        
        best_result_str = (
            f"Best Model Saved: best_dice.pth\n"
            f"miou: {best_metrics['miou']}, dice: {best_metrics['dice']}, "
            f"accuracy: {best_metrics['accuracy']}, sensitivity: {best_metrics['sensitivity']}, "
            f"specificity: {best_metrics['specificity']}\n"
            "----------------------------------------"
        )
        
        print(best_result_str)
        performance_logger.info(best_result_str)
if __name__ == "__main__":
    from build_model import build_model

    # l=['mysyn']
    save_idx=200
    l = [str(i) for i in range(0, save_idx)]
    l.append('best_loss')     
    name="base0.0001"
    model=build_model(model=name,num_classes=1)
    dataset='isic2018'
    print(name,dataset)
    name=dataset+'_'+name
    test(model=model, l=l, name=name,dataset=dataset,results_dir='./results/isic')