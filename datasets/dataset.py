from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as F
import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
from torchvision import transforms
class NPY_datasets(Dataset):
    def __init__(self, path_Data, train=True,dataset=None):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                    myNormalize(data_name=dataset,train=True),
                    myToTensor(),
                    myRandomHorizontalFlip(p=0.5),
                    myRandomVerticalFlip(p=0.5),
                    myRandomRotation(p=0.5, degree=[0, 360]),
                    myResize(256, 256)   # 归一化        
                        ])
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                        myNormalize(data_name=dataset,train=False),
                        myToTensor(),
                        myResize(256, 256)
                    ])
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
    


    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
       

class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return F.resize(image, [self.size_h, self.size_w],antialias=True), F.resize(mask, [self.size_h, self.size_w],antialias=True)
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return F.hflip(image), F.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return F.vflip(image), F.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return F.rotate(image,self.angle), F.rotate(mask,self.angle)
        else: return image, mask 
class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic2018':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic2017':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
        else:
            self.mean = 159.922
            self.std = 28.871

            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk

class normalNormalize:
    def __init__(self, data_name=None, train=True):
        self.mean = [123.675, 116.28, 103.53]  
        self.std = [58.395, 57.12, 57.375]    
        
    def __call__(self, data):
        img, msk = data
        
        if len(img.shape) == 2:  
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1: 
            img = np.repeat(img, 3, axis=-1)
            
        img_normalized = (img - self.mean) / self.std
        img_normalized = img_normalized.astype(np.float32)
        
        return img_normalized, msk
class other_datasets(Dataset):
    def __init__(self, path_Data, train=True,dataset=None,image_size=256):
        super(other_datasets, self)
        self.image_size = image_size
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                    toResize(self.image_size, self.image_size),
                    myNormalize(data_name=dataset,train=True),
                    myToTensor(),
                    myRandomHorizontalFlip(p=0.5),
                    myRandomVerticalFlip(p=0.5),
                    myRandomRotation(p=0.5, degree=[0, 360]),
                        ])
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                        toResize(self.image_size, self.image_size),
                        myNormalize(data_name=dataset,train=False),
                        myToTensor(),
                    ])
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img_path, msk_path = self.data[indx]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        img, msk = self.transformer((img, msk))
        msk=np.where(msk > 0, 1, 0)
        return img, msk

    def __len__(self):
        return len(self.data)

class cam_datasets(Dataset):
    def __init__(self, path_Data, train=True, dataset=None, image_size=256):
        super(cam_datasets, self).__init__()
        self.image_size = image_size
        if train:
            images_list = sorted(os.listdir(os.path.join(path_Data, 'train', 'images')))
            masks_list = sorted(os.listdir(os.path.join(path_Data, 'train', 'masks')))
            self.data = []
            for i in range(len(images_list)):
                img_path = os.path.join(path_Data, 'train', 'images', images_list[i])
                mask_path = os.path.join(path_Data, 'train', 'masks', masks_list[i])
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                    toResize(self.image_size, self.image_size),
                    myNormalize(data_name=dataset, train=True),
                    myToTensor(),
                    myRandomHorizontalFlip(p=0.5),
                    myRandomVerticalFlip(p=0.5),
                    myRandomRotation(p=0.5, degree=[0, 360]),
                ])
        else:
            images_list = sorted(os.listdir(os.path.join(path_Data, 'val', 'images')))
            masks_list = sorted(os.listdir(os.path.join(path_Data, 'val', 'masks')))
            self.data = []
            for i in range(len(images_list)):
                img_path = os.path.join(path_Data, 'val', 'images', images_list[i])
                mask_path = os.path.join(path_Data, 'val', 'masks', masks_list[i])
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                    toResize(self.image_size, self.image_size),
                    myNormalize(data_name=dataset, train=False),
                    myToTensor(),
                ])
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)

class toResize:
    def __init__(self, height, width):
        self.size = (height, width)
        
    def __call__(self, sample):
        image, mask = sample
        
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, interpolation=F.InterpolationMode.NEAREST)
        
        image = np.array(image)
        mask = np.array(mask)[..., np.newaxis] 
        
        return image, mask