import os
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
from torch import optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import deeplake
import albumentations as A
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import random

# Data saving dir path
os.environ["DEEPLAKE_DOWNLOAD_PATH"] = "~/data/"

# Data2#class mapping
data_cls_map = {
    "cifar10" : 10,
    "cifar100" : 100,
    "imagenet" : 1000
}

# Data Augmentation Callable Function
transform = A.Compose([
    A.RGBShift(),
    A.RandomBrightnessContrast(),
    A.GaussianBlur(),
    A.HorizontalFlip(),
    A.SafeRotate(),
    A.ColorJitter(),
    A.Normalize()
])

# Classification Data set from Deep Lake
class CLFDataset(Dataset):
    def __init__(self, args: argparse, subset = "train") -> None:
        super().__init__()
        
        self.args = args
        self.subset = subset
        self.ds = deeplake.load(
            path = f'hub://activeloop/{args.dataset}-{self.subset}',
            access_method = 'local')

    def __getitem__(self, index):        
        img = self.ds[index]["images"].data()["value"]
        label = self.ds[index]["labels"].data()["value"].tolist()[0]
        processed_imgs = torch.from_numpy(transform(image = img)["image"]).permute(-1,0,1)   
         
        return (processed_imgs, label)
    
    def __len__(self):
        return len(self.ds)


# Key2Model mapping
model_dict = {
    "resnet18": models.resnet18,
    "resnet34" : models.resnet34,
    "resnet50": models.resnet50,
    "effb0" : models.efficientnet_b0
}

# LARS Optimizer
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def main(args: argparse):
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{args.port}'
    args.world_size = args.ngpus_per_node
    mp.spawn(main_worker, (args,), args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    
    dist.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    
    if args.rank == 0:
        log = {
            "train_loss" : [],
            "train_acc" : [],
            "test_loss" : [],
            "test_acc" : []
        }
        
        log_path = os.getcwd() + f"/{args.dataset}-{args.model}-{args.bs}.parquet"
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # Model
    model = model_dict[args.model](num_classes = data_cls_map[args.dataset]).cuda(gpu)
    model = DDP(model, device_ids=[gpu])
    
    # Optimizer
    optimizer = LARS(
        model.parameters(), lr=args.lr, weight_decay=0, 
        weight_decay_filter=True, lars_adaptation_filter=True
    )
    
    # Dataset
    train_dataset = CLFDataset(args=args, subset='train')
    test_dataset = CLFDataset(args=args, subset='test')
    assert args.bs % args.world_size == 0
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    per_device_batch_size = args.bs // args.world_size
    
    # Data Loader
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=test_sampler
    )
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        correct = 0 
        _loss = 0
        for step, (train_img, train_label) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            train_img = train_img.cuda(gpu, non_blocking=True)
            train_label = train_label.cuda(gpu, non_blocking=True)
            logits = model(train_img)
            loss = criterion(logits, train_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.rank == 0:
                correct += (logits.argmax(1) == train_label).type(torch.float).sum().item()
                _loss += loss.item()
        
        if args.rank == 0:
            log["train_loss"].append(_loss/len(train_loader))
            log["train_acc"].append(correct/len(train_loader.dataset))
        
        test_sampler.set_epoch(epoch)
        with torch.no_grad():
            correct = 0 
            _loss = 0
            for _, (val_img, val_label) in tqdm(enumerate(test_loader)):
                val_img = val_img.cuda(gpu, non_blocking=True)
                val_label = val_label.cuda(gpu, non_blocking=True)
                logits = model(val_img)
                loss = criterion(logits, val_label)
            
                if args.rank == 0:
                    correct += (logits.argmax(1) == val_label).type(torch.float).sum().item()
                    _loss += loss.item()
            
            if args.rank == 0:
                log["test_loss"].append(_loss/len(test_loader))
                log["test_acc"].append(correct/len(test_loader.dataset))
        
        if args.rank == 0:
            print(f"Epoch: {epoch} - " + " - ".join([f"{key}: {log[key][epoch]}" for key in log]))
    
    if args.rank == 0:
        log_df = pd.DataFrame(log)
        log_df.to_parquet(log_path)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Torch Multi-GPU Baseline',
                    description='This project conduct the benchmark of among batch sizes in multi-gpu',
                    epilog='ENJOY!!!')
    
    parser.add_argument('--dataset', type = str, default='cifar100',
                    help='dataset name', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--bs', type = int, default=32,
                    help='batch size')
    parser.add_argument('--workers', type = int, default=4,
                    help='Number of processor used in data loader')
    parser.add_argument('--model', type = str, default='effb0',
                    help='model name', choices=['resnet18', 'resnet50', 'resnet34', 'effb0'])
    parser.add_argument('--epochs', type = int, default=1,
                    help='# Epochs used in training')
    parser.add_argument('--lr', type=float, default=0.001, 
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=777, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--port', type=int, default=8080, help='Multi-GPU Training Port.')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args=args)