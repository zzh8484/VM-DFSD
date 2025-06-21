import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset import Synapse_dataset, RandomGenerator
import torch.optim as optim
import os
import random
import logging
import time
import numpy as np
import torch.backends.cudnn as cudnn
from utils.myutils import  *
from torch.nn.modules.loss import CrossEntropyLoss
from syn_mytest import *
from build_model import build_model
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

def prepare_dataloader(batch_size=16,base_dir=None,list_dir=None):
    train_dataset = Synapse_dataset(
        base_dir=base_dir,
        list_dir=list_dir,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[224, 224])
        ])
    )
    
    return DataLoader( 
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        dataset=train_dataset
    )
def train(args):
    model = build_model(model=args.name,num_classes=9,scan_mode=args.scan,k_group=args.group,pretain=args.pretain)
    print(device)
    train_loader = prepare_dataloader(batch_size=args.batch_size,base_dir=args.base_dir,list_dir=args.list_dir)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-2, 
        amsgrad=False
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.T_max,        
        eta_min=1e-5,    
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = nDiceLoss(9)
    dir = args.results_dir
    dir_name = os.path.join(dir, args.name)
    save_idx=0
    w_ce, w_dice = 0.5, 0.5
    best_loss = float('inf')  

    for epoch in range(args.epoch+1):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for data in train_loader:
            images = data['image'].to(device, non_blocking=True)
            labels = data['label'].squeeze(1).to(device, non_blocking=True)

            
            outputs = model(images)
            loss_ce = ce_loss(outputs, labels[:].long())
            loss_dice = dice_loss(outputs, labels, softmax=True)
            loss = (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        now_lr = optimizer.param_groups[0]['lr'] 

        scheduler.step() 
        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(os.path.join(dir_name, "pth"), exist_ok=True)
        if now_lr <=1e-5:
            idx_path = os.path.join(dir_name, f'pth/{save_idx}.pth')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                }, 
                idx_path
            )
            print(f"Epoch {epoch} | Model saved with loss: {avg_loss:.4f}")
            save_idx+=1
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(dir_name, f'pth/best_lose.pth')
            torch.save(
                {
                    'model_state_dict': model.state_dict()
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
    test(name=args.name,model=model,results_dir=dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='base',type=str)
    parser.add_argument('--scan', default=3,type=int)
    parser.add_argument('--group', default=8,type=int)
    parser.add_argument('--pretain', default='True',type=str)
    parser.add_argument('--epoch', default=250,type=int)
    parser.add_argument('--seed', default=42,type=int)
    parser.add_argument('--lr', default=0.001,type=float)
    parser.add_argument('--T_max', default=50,type=int)
    parser.add_argument('--batch_size', default=16,type=int)
    parser.add_argument('--base_dir', default='data/Synapse/train_npz',type=str)
    parser.add_argument('--list_dir', default='data/lists_Synapse',type=str)
    parser.add_argument('--results_dir', default='results/syn',type=str)
    args = parser.parse_args()
    set_seed(args.seed)
    train(args=args)

