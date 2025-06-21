import os
import torch
import torch.optim as optim
import logging
from torch.optim import lr_scheduler
import time
import numpy as np
import torch.backends.cudnn as cudnn
from utils.myutils import  *
from datasets.dataset import *
import random
import logging
import time
import torch.backends.cudnn as cudnn
from isic_mytest import *
from build_model import *
import argparse

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def prepare_dataloader(batch_size=32,dataset="isic2017"):
    train_dataset = NPY_datasets(
        path_Data='/root/haiguang/zzh/datasets/'+dataset+'/',
        train=True,
        dataset=dataset
    )
    
    return DataLoader( 
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        dataset=train_dataset
    )


def train(args=None):
    model = build_model(model=args.name,num_classes=1,scan_mode=args.scan,k_group=args.group,device=device,pretain=args.pretain)
    train_loader = prepare_dataloader(dataset=args.dataset,batch_size=args.batch_size)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-2, 
        amsgrad=False
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.T_max,        
        eta_min=1e-5,   
    )
    bc_loss = BCELoss()
    dice_loss = DiceLoss()
    dir = args.results_dir
    name=args.dataset+"_"+args.name
    dir_name = os.path.join(dir,name)
    w_ce, w_dice = 0.5, 0.5
    save_idx = 0
    best_loss = float('inf')  
    for epoch in range(args.epoch+1):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        start_time = time.time()
        for data in train_loader:
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            
            outputs = model(images)
            loss_ce = bc_loss(outputs, targets)
            loss_dice = dice_loss(outputs, targets)
            loss = (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        now_lr = optimizer.param_groups[0]['lr'] 

        scheduler.step() 
        if now_lr <= 1e-5:
            idx_path = os.path.join(dir_name, f'pth/{save_idx}.pth')
            torch.save(
                {
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                }, 
                idx_path
            )
            save_idx+=1
        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(os.path.join(dir_name, "pth"), exist_ok=True)
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(dir_name, f'pth/best_loss.pth')
            torch.save(
                {
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                }, 
                best_path
            )
            print(f"Epoch {epoch} | Best model saved with loss: {avg_loss:.4f}")
        etime = time.time()
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Sum: {total_loss:.2f} | LR: {now_lr} | Time: {etime-start_time:.2f}s")
    model_logger = create_logger(
        logger_name="model",
        filename=os.path.join(dir_name, 'pth/model.log'),
        level=logging.DEBUG,
        add_console=False, 
    )
    model_logger.info(model)
    test(name=name,model=model,dataset=args.dataset,results_dir=args.results_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='isic2018',type=str, help='datsets')
    parser.add_argument('--name', default='base',type=str)
    parser.add_argument('--scan', default=3,type=int)
    parser.add_argument('--group', default=8,type=int)
    parser.add_argument('--pretain', default='True',type=str)
    parser.add_argument('--epoch', default=50,type=int)
    parser.add_argument('--seed', default=42,type=int)
    parser.add_argument('--lr', default=0.001,type=float)
    parser.add_argument('--T_max', default=50,type=int)
    parser.add_argument('--batch_size', default=32,type=int)
    parser.add_argument('--results_dir', default='results/isic',type=str)
    args = parser.parse_args()

    set_seed(args.seed)
    train(args=args)