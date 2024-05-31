import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils import cluster_acc, Identity, AverageMeter, seed_torch, initialize
from models.resnet import ResNet, BasicBlock, Bottleneck, Classifier, Projector, resnet18, resnet50
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from data.svhnloader import SVHNLoader
from data.imagenetloader import ImageNetLoader882_30Mix_pre, ImageNetLoader30_pre, ImageNetLoader882_pre, ImageNetLoader30, ImageNetLoader882
from data.cubloader import get_cub_datasets, CustomCub2011
from tqdm import tqdm
import numpy as np
import os
from termcolor import colored
from kmeans.kmeans_pytorch import kmeans, kmeans_predict
from SimCLR.simclr import SimCLR
from SimCLR.simclr.modules.identity import Identity

def lr_sche(eta, factor, epoch):
    discount = epoch//factor + 1
    return eta**discount

def train(model, train_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        record = []
        for idx, (x, label, index) in enumerate(tqdm(train_loader)):
            x, label = x.to(device), label.to(device)
            feat, output = model(x)
            loss = criterion1(output, label)
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        exp_lr_scheduler.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        accuracy_tr = cls_test(model, mix_train_loader, args)

def cls_test(model, test_loader, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label, index in tqdm(test_loader):
            x, label = x.to(device), label.to(device)
            feat, output = model(x)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the model on the noisy data: %d %%' % (accuracy))
    return accuracy

def meta_KMeans(model, test_loader, args):
    accs, nmis, aris = np.array([]), np.array([]), np.array([])
    
    for epoch in range(args.epochs_km):
        cluster_centers_record = []
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x = x.to(device)
            feat, output = model(x)
            
            if epoch == 0 and batch_idx == 0:
                avg_centers = []
            elif epoch == 0 and batch_idx >0:
                avg_centers = cluster_centers
            
            cluster_idx, cluster_centers = kmeans(
                X=output, num_clusters=args.num_unlabeled_classes, distance='euclidean',
                cluster_centers=avg_centers,
                tol=1e-4, device=torch.device('cuda'))
            cluster_centers_record.append(cluster_centers.cpu().detach().numpy())

        avg_centers = (1-lr_sche(args.eta,args.factor,epoch))*avg_centers + lr_sche(args.eta,args.factor,epoch)*torch.mean(torch.tensor(cluster_centers_record), dim=0)
        acc, nmi, ari = test(model, avg_centers.detach().numpy(), test_loader, args)
        accs = np.append(accs, acc); nmis = np.append(nmis, nmi); aris = np.append(aris, ari)

def test(model, cluster_centers, test_loader, args):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        feat, output = model(x)
        pred = kmeans_predict(output, torch.from_numpy(cluster_centers), 'euclidean', device=torch.device('cuda:0'))
        preds = np.append(preds, pred.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc, nmi, ari

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--epochs_km', default=50, type=int)
    parser.add_argument('--step_size', default=40, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='your data dir')
    parser.add_argument('--exp_root', type=str, default='your results saving dir')
    parser.add_argument('--resnet', type=str, default='resnet18')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, imagenet')
    parser.add_argument('--unlabeled_subset', type=str, default='A')
    parser.add_argument('--unlabeled_batch_size', type=int, default=128)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--factor', type=int, default=100)
    parser.add_argument('--noise_type', type=str, default='ncd_noisify')
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--cross_rate', type=float, default=1.0)
    parser.add_argument('--encoder_dir', type=str, default="you encoder dir")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}_{}_{}.pth'.format(args.dataset_name, args.noise_rate, args.cross_rate)
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    
    if args.dataset_name == 'cifar10':
        mix_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', num_workers=8, aug=None, shuffle=True, target_list=range(num_classes), noise_type=args.noise_type, noise_rate=args.noise_rate, cross_rate=args.cross_rate, random_state=0)
        mix_test_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', num_workers=8, aug=None, shuffle=False, target_list=range(num_classes), noise_type=args.noise_type, noise_rate=args.noise_rate, cross_rate=args.cross_rate, random_state=0)
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes), noise_type='clean', noise_rate=None, cross_rate=None, random_state=None)
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=True, target_list = range(args.num_labeled_classes, num_classes), noise_type='clean', noise_rate=None, cross_rate=None, random_state=None)
    elif args.dataset_name == 'cifar100':
        mix_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', num_workers=8, aug=None, shuffle=True, target_list=range(num_classes), noise_type=args.noise_type, noise_rate=args.noise_rate, cross_rate=args.cross_rate, random_state=0)
        mix_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', num_workers=8, aug=None, shuffle=False, target_list=range(num_classes), noise_type=args.noise_type, noise_rate=args.noise_rate, cross_rate=args.cross_rate, random_state=0)
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes), noise_type='clean', noise_rate=None, cross_rate=None, random_state=None)
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=True, target_list = range(args.num_labeled_classes, num_classes), noise_type='clean', noise_rate=None, cross_rate=None, random_state=None)
    elif args.dataset_name == 'svhn':
        mix_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', num_workers=8, aug=None, shuffle=True, target_list=range(num_classes), noise_type=args.noise_type, noise_rate=args.noise_rate, cross_rate=args.cross_rate, random_state=0)
        mix_test_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', num_workers=8, aug=None, shuffle=False, target_list=range(num_classes), noise_type=args.noise_type, noise_rate=args.noise_rate, cross_rate=args.cross_rate, random_state=0)
        unlabeled_eval_loader_test = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=True, target_list = range(args.num_labeled_classes, num_classes), noise_type='clean', noise_rate=None, cross_rate=None, random_state=None)
        unlabeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=True, target_list = range(args.num_labeled_classes, num_classes), noise_type='clean', noise_rate=None, cross_rate=None, random_state=None)
    elif args.dataset_name == 'imagenet':
        mix_train_loader = ImageNetLoader882_30Mix_pre(args.batch_size, num_workers=16, path=args.dataset_root, 
                                                       unlabeled_subset=args.unlabeled_subset, aug='twice_pre', 
                                                       shuffle=True, subfolder='train', unlabeled_batch_size=args.unlabeled_batch_size,
                                                       noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=0)
        labeled_eval_loader = ImageNetLoader882(args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, subfolder='val')
        unlabeled_eval_loader = ImageNetLoader30(args.batch_size, num_workers=16, path=args.dataset_root, subset=args.unlabeled_subset, aug=None, shuffle=False, subfolder='train')
    elif args.dataset_name == 'cub':
        get_cub_datasets

    if args.resnet == 'resnet50':
        encoder = resnet50(pretrained=False)
        n_features = 512 * 4
    elif args.resnet == 'resnet18':
        encoder = resnet18(pretrained=False)
        n_features = 512 * 1
    else:
        raise NotImplementedError
    
    encoder.load_state_dict(torch.load(args.encoder_dir, map_location='cuda'), strict=False)
    encoder.to(device)

    if args.dataset_name == 'imagenet':
        moco_state = torch.load('r-50-1000ep.pth.tar', map_location='cpu')

        # Transfer moco weights
        print(colored('Transfer MoCo weights to model', 'blue'))
        new_state_dict = {}
        state_dict = moco_state['state_dict']
        
        for k in list(state_dict.keys()):
            # Copy backbone weights
            if k.startswith('module.momentum_encoder') and not k.startswith('module.momentum_encoder.fc'):
                new_k = 'module.encoder.' + k[len('module.momentum_encoder.'):]
                new_state_dict[new_k] = state_dict[k]
            
            # Copy mlp weights
            elif k.startswith('module.momentum_encoder.fc'):
                new_k = 'module.projector.' + k[len('module.momentum_encoder.fc.'):] 
                new_state_dict[new_k] = state_dict[k]
            elif k.startswith('module.base_encoder'):
                continue
            elif k.startswith('module.predictor'):
                continue
            else:
                raise ValueError('Unexpected key {}'.format(k))

    if args.mode == 'train':
        classifier = Classifier(encoder, n_features, args.num_labeled_classes+1).to(device)
        train(classifier, mix_train_loader, args)
        torch.save(classifier.encoder.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
        meta_KMeans(classifier.encoder, unlabeled_eval_loader, args)
    else:
        meta_KMeans(encoder, unlabeled_eval_loader, args)