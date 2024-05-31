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

import random
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity, ncd_noisify
import torchvision.transforms as transforms

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}
    known_class = [i for i in range(5)]
    novel_class = [i for i in range (5,10)]

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=True, target_list=range(5),
                 noise_type=None, noise_rate=0.2, cross_rate=0.5, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set
        self.noise_type = noise_type
        self.target_list = target_list

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        ind = [i for i in range(len(self.labels)) if int(self.labels[i]) in target_list]
        
        self.data = self.data[ind]
        self.labels= self.labels[ind]

        if noise_type != 'clean':
            self.train_labels = np.asarray([[self.labels[i]] for i in range(len(self.labels))])
            self.noisy_labels, self.actual_noise_rate = \
                ncd_noisify(y_train=self.train_labels, noise_rate=noise_rate, cross_rate=cross_rate, random_state=random_state, \
                            num_labeled_classes=len(self.known_class), num_unlabeled_classes=len(self.novel_class))
            self.noisy_labels=[i[0] for i in self.noisy_labels]
            _train_labels=[i[0] for i in self.train_labels]
            self.noise_or_not = np.transpose(self.noisy_labels)==np.transpose(_train_labels)

            self.ncd_noisy_labels = [i if i<len(self.known_class) else len(self.known_class) for i in self.noisy_labels]

            ind_known = [i for i in range(len(self.noisy_labels)) if self.noisy_labels[i] in self.known_class]
            ind_novel = [i for i in range(len(self.noisy_labels)) if self.noisy_labels[i] in self.novel_class]
            assert len(ind_known)+len(ind_novel)==len(self.labels)

            self.data_known = self.data[ind_known]
            self.data_novel = self.data[ind_novel]
            self.labels = np.array(self.labels)
            self.noisy_labels = np.array(self.noisy_labels)
            self.labels_known = self.noisy_labels[ind_known].tolist()
            self.labels_novel = self.noisy_labels[ind_novel].tolist()
            self.labels_known_gd = self.labels[ind_known].tolist()
            self.labels_novel_gd = self.labels[ind_novel].tolist()

        elif noise_type == 'clean':
            ind_known = [i for i in range(len(self.labels)) if self.labels[i] in self.known_class]
            ind_novel = [i for i in range(len(self.labels)) if self.labels[i] in self.novel_class]
            assert len(ind_known)+len(ind_novel)==len(self.labels)

            self.data_known = self.data[ind_known]
            self.data_novel = self.data[ind_novel]
            self.labels = np.array(self.labels)
            self.labels_known = self.labels[ind_known].tolist()
            self.labels_novel = self.labels[ind_novel].tolist()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #img, target = self.data[index], int(self.labels[index])
        if self.target_list == range(5):
            img, target = self.data_known[index], int(self.labels_known[index])
        elif self.target_list == range(5,10):
            img, target = self.data_novel[index], int(self.labels_novel[index])
        elif self.target_list == range(10) and self.noise_type != 'clean':
            img, target = self.data[index], int(self.ncd_noisy_labels[index])
        elif self.target_list == range(10) and self.noise_type == 'clean':
            img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.target_list == range(5):
            return len(self.data_known)
        elif self.target_list == range(5,10):
            return len(self.data_novel)
        elif self.target_list == range(10):
            return len(self.data)
    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def SVHNData(root, split='train', aug=None, target_list=range(5), noise_type=None, noise_rate=0.5, cross_rate=0.5, random_state=0):
    if aug==None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  
    elif aug=='once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])  
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]))
    dataset = SVHN(root=root, split=split, transform=transform, target_list=target_list, noise_type=noise_type, noise_rate=noise_rate, cross_rate=cross_rate, random_state=random_state)
    return dataset

def SVHNLoader(root, batch_size, split='train', num_workers=2,  aug=None, shuffle=True, target_list=range(5), noise_type=None, noise_rate=0.2, cross_rate=0.5, random_state=0):
    dataset = SVHNData(root, split, aug, target_list, noise_type, noise_rate, cross_rate, random_state)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader