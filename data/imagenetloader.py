from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader
from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler, ConcatDataset, ncd_noisify

def find_classes_from_folder(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def find_classes_from_file(file_path):
    with open(file_path) as f:
            classes = f.readlines()
    classes = [x.strip() for x in classes] 
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, classes, class_to_idx):
    samples = []
    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            print("Class {} does not exit".format(target))
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                if 'JPEG' in path or 'jpg' in path:
                    samples.append(item)
    
    return samples 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder_Noisy(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None,
                 noise_type=None, noise_rate=0.2, cross_rate=0.3, random_state=0, loader=pil_loader):
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 images in subfolders \n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.samples=samples 
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        known_idx = [i for i in range(882)]
        novel_idx = [i for i in range(882,882+30)]

        self.data_path = list(zip(*self.samples))[0]
        self.targets = list(zip(*self.samples))[1]
        
        if noise_type != 'clean':
            self.train_labels = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
            self.noisy_labels, self.actual_noise_rate = \
                ncd_noisify(y_train=self.train_labels, noise_rate=noise_rate, cross_rate=cross_rate, random_state=random_state, num_labeled_classes=len(known_idx), num_unlabeled_classes=len(novel_idx))
            self.noisy_labels=[i[0] for i in self.noisy_labels]
            _train_labels=[i[0] for i in self.train_labels]
            self.noise_or_not = np.transpose(self.noisy_labels)==np.transpose(_train_labels)
            
            ind_known = [i for i in range(len(self.noisy_labels)) if self.noisy_labels[i] in known_idx]
            ind_novel = [i for i in range(len(self.noisy_labels)) if self.noisy_labels[i] in novel_idx]
            assert len(ind_known)+len(ind_novel)==len(self.targets)
            self.num_known = len(ind_known)
            self.num_novel = len(ind_novel)
            self.data_path = np.array(self.data_path)
            self.path_known = self.data_path[ind_known]
            self.path_novel = self.data_path[ind_novel]
            self.data_path = np.concatenate((self.path_known,self.path_novel))
            self.noisy_labels = np.array(self.noisy_labels)
            self.targets_known = self.noisy_labels[ind_known]
            self.targets_novel = self.noisy_labels[ind_novel]
            self.targets = np.concatenate((self.targets_known,self.targets_novel)).tolist()
        
    def __getitem__(self, index):
        path = self.data_path[index]
        target = self.targets[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.targets)

class ImageFolder(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None, loader=pil_loader):
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 images in subfolders \n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.samples=samples 
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)


def ImageNet_all(aug=None, subfolder='train', path='./data/datasets/ImageNet/', subset='A', noise_type='symmetric', noise_rate=0.2, random_state=0):
    img_split = 'images/'+subfolder
    classes_118, class_to_idx_118 = find_classes_from_file(os.path.join(path, 'imagenet_rand118/imagenet_118.txt'))
    samples_118 = make_dataset(path+img_split, classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(os.path.join(path, img_split))
    classes_882 = list(set(classes_1000) - set(classes_118))
    class_to_idx_882 = {classes_882[i]: i for i in range(len(classes_882))}
    classes_30, class_to_idx_30 = find_classes_from_file(os.path.join(path, 'imagenet_rand118/imagenet_30_{}.txt'.format(subset)))
    classes_all = classes_882 + classes_30
    class_to_idx_all = {classes_all[i]: i for i in range(len(classes_all))}
    samples_all = make_dataset(path+img_split, classes_all, class_to_idx_all)

    if aug==None:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    if aug=='none_pre':
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once_pre':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='twice_pre':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='ktimes':
        transform = TransformKtimes(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), k=10)
    dataset = ImageFolder_Noisy(transform=transform, samples=samples_all,noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
    return dataset 

def ImageNet882(aug=None, subfolder='train', path='./data/datasets/ImageNet/', noise_type='clean', noise_rate=0, random_state=0):
    img_split = 'images/'+subfolder
    classes_118, class_to_idx_118 = find_classes_from_file(os.path.join(path, 'imagenet_rand118/imagenet_118.txt'))
    samples_118 = make_dataset(path+img_split, classes_118, class_to_idx_118)
    classes_1000, _ = find_classes_from_folder(os.path.join(path, img_split))
    classes_882 = list(set(classes_1000) - set(classes_118))
    class_to_idx_882 = {classes_882[i]: i for i in range(len(classes_882))}
    samples_882 = make_dataset(path+img_split, classes_882, class_to_idx_882)
    if aug==None:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    if aug=='none_pre':
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once_pre':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='twice_pre':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='ktimes':
        transform = TransformKtimes(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), k=10)
    dataset = ImageFolder(transform=transform, samples=samples_882)
    return dataset 

def ImageNet30(path='./data/datasets/ImageNet/', subset='A', aug=None, subfolder='train', noise_type='clean', noise_rate=0, random_state=0):
    classes_30, class_to_idx_30 = find_classes_from_file(os.path.join(path, 'imagenet_rand118/imagenet_30_{}.txt'.format(subset)))
    samples_30 = make_dataset(path+'images/{}'.format(subfolder), classes_30, class_to_idx_30)
    if aug==None:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='none_pre':
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
    elif aug=='once':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='once_pre':
        transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='twice_pre':
        transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug=='ktimes':
        transform = TransformKtimes(transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), k=10)

    dataset = ImageFolder(transform=transform, samples=samples_30)
    return dataset

def ImageNetLoader30(batch_size, num_workers=2, path='./data/datasets/ImageNet/', subset='A', aug=None, shuffle=False, subfolder='train'):
    dataset = ImageNet30(path, subset, aug, subfolder)
    dataloader_30 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) 
    return dataloader_30


def ImageNetLoader30_pre(batch_size, num_workers=2, path='./data/datasets/ImageNet/', subset='A', aug=None, shuffle=False, subfolder='train'):
    dataset = ImageNet30(path, subset, aug, subfolder)
    dataloader_30 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=fast_collate)
    return dataloader_30


def ImageNetLoader882(batch_size, num_workers=2, path='./data/datasets/ImageNet/', aug=None, shuffle=False, subfolder='train'):
    dataset = ImageNet882(aug=aug, subfolder=subfolder, path=path)
    dataloader_882 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader_882

def ImageNetLoader882_pre(batch_size, num_workers=2, path='./data/datasets/ImageNet/', aug=None, shuffle=False, subfolder='train'):
    dataset = ImageNet882(aug=aug, subfolder=subfolder, path=path)
    dataloader_882 = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=fast_collate)
    return dataloader_882

def ImageNetLoader882_30Mix(batch_size, num_workers=2, path='./data/datasets/ImageNet/',  unlabeled_subset='A', aug=None, shuffle=False,\
                            subfolder='train', unlabeled_batch_size=64, noise_type='symmetric', noise_rate=0.2, random_state=0):
    dataset = ImageNet_all(aug=aug, subfolder=subfolder, path=path, subset=unlabeled_subset, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
    #dataset_labeled = ImageNet882(aug=aug, subfolder=subfolder, path=path)
    #dataset_unlabeled = ImageNet30(path, unlabeled_subset, aug, subfolder)
    #dataset= ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(dataset.num_known)
    unlabeled_idxs = range(dataset.num_known, dataset.num_known+dataset.num_novel)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size) 
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True)
    loader.labeled_length = dataset.num_known
    loader.unlabeled_length = dataset.num_novel
    return loader

def ImageNetLoader882_30Mix_pre(batch_size, num_workers=2, path='./data/datasets/ImageNet/',  unlabeled_subset='A', aug=None, shuffle=False, \
                                subfolder='train', unlabeled_batch_size=64, noise_type='symmetric', noise_rate=0.2, random_state=0):
    dataset = ImageNet_all(aug=aug, subfolder=subfolder, path=path, subset=unlabeled_subset, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
    labeled_idxs = range(dataset.num_known)
    unlabeled_idxs = range(dataset.num_known, dataset.num_known+dataset.num_novel)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size)
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=fast_collate2, pin_memory=True)
    loader.labeled_length = dataset.num_known
    loader.unlabeled_length = dataset.num_novel
    return loader


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    idxs = torch.tensor([idx[2] for idx in batch], dtype=torch.int64)
    w = imgs[0].size()[1]
    h = imgs[0].size()[2]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        #nump_array = np.transpose(nump_array, (1,2,0))
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        #nump_array = np.rollaxis(nump_array, 2, 0)
        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets, idxs


def fast_collate2(batch):
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    idxs = torch.tensor([idx[2] for idx in batch], dtype=torch.int64)

    imgs = [img[0][0] for img in batch]

    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)

    imgs1 = [img[0][1] for img in batch]
    w = imgs1[0].size[0]
    h = imgs1[0].size[1]
    tensor1 = torch.zeros((len(imgs1), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs1):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor1[i] += torch.from_numpy(nump_array)

    return tensor, tensor1, targets, idxs
