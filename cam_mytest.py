
import numpy as np
import os
from tqdm import tqdm
from scipy.ndimage import zoom
from medpy import metric
from datasets.dataset import *
from torch.utils.data import DataLoader
from lib.networks import *
import logging
from utils.myutils import *
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 =0
        jaccard = 0
        asd = 0
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def prepare_dataloader_val():
    dataset="Echocardiography-CAMUS"
    val_dataset = cam_datasets(
        path_Data='data/US30K/' + dataset + '/',
        train=False,
        dataset=dataset,
        image_size=256
    )

    return   DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4, )

def test_single_volume(image, label, net, classes, patch_size=[256, 256],only_dice=True):
    image, label = image, label.squeeze(0).cuda().cpu().numpy()
    out = torch.argmax(torch.softmax(net(image), dim=1), dim=1).squeeze(0)
    prediction = out.cpu().detach().numpy()

    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

def inference(model,testloader,label=None,only_dice=True):  
    model.eval()
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):       
        image, label = sampled_batch
        image, label = image.cuda(non_blocking=True).float(), label.cuda(non_blocking=True).long()
        metric_i = test_single_volume(image, label, model, classes=4, patch_size=[256, 256],only_dice=only_dice)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(testloader)
    performance = np.mean(metric_list, axis=0)[0]
    return performance,metric_list




def test(name=None,model=None,results_dir=None):
    name = 'Echocardiography-CAMUS' + "_" + name

    pth_dir = os.path.join(results_dir, name, "pth")
    log_dir = os.path.join(results_dir, name, "log")
    os.makedirs(log_dir, exist_ok=True)  
    performance_logger = create_logger(
    logger_name="performance",
    filename=os.path.join(log_dir, "performance.log"),  
    level=logging.INFO,
    filemode='a',
    add_console=True
    )
    testloader = prepare_dataloader_val()
    model.eval()
    best_dice = 0.0  
    best_checkpoint_path = None  

    for filename in os.listdir(pth_dir):
        if not filename.endswith('.pth'):
            continue
        print(f"Processing: {filename}")
        checkpoint = torch.load(
            os.path.join(pth_dir, filename), 
            map_location=torch.device('cpu')
        )
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)

        performance, class_metric = inference(model, testloader,only_dice=True)

        performance_logger.info(f"Model: {filename}")
        performance_logger.info(f"Overall Dice: {performance*100:.2f}")

        
        if performance > best_dice:
            best_dice = performance
            best_checkpoint_path = os.path.join(pth_dir, filename)
    
    if best_checkpoint_path:
        import shutil
        good_model_path = os.path.join(pth_dir, "best_dice.pth")
        shutil.copyfile(best_checkpoint_path, good_model_path)
        performance, class_metric= inference(model, testloader,only_dice=False)
        performance_logger.info(f"Best model saved: {good_model_path}, Dice: {best_dice*100:.2f}")
        for idx, dice in enumerate(class_metric, 1):
            performance_logger.info(f"Class {idx} Dice: {dice[0]*100:.2f},hd5:  {dice[1]:.2f},jaccard: {dice[2]*100:.2f},asd: {dice[3]:.2f}")



if __name__ == "__main__":


    from build_model import build_model
    model=build_model(model='base',num_classes=4,scan_mode=3,k_group=8)
    test(model=model,name='base',results_dir='results/other')