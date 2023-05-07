import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings('ignore', category = UserWarning)
import torch
from torch import nn
from torch import optim
import torchvision.models as models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import random

# LARS Optimizer
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=5e-4, momentum=0.9, eta=0.001,
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
        
        log_path = os.getcwd() + f"/{args.bs}_{args.lr}.parquet"
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # Model
    model = models.resnet18(num_classes = 10).cuda(gpu)
    model = DDP(model, device_ids=[gpu])
    
    # Optimizer
    optimizer = LARS(
        model.parameters(), lr=args.lr, weight_decay_filter=True, lars_adaptation_filter=True
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Dataset
    train_dataset = datasets.CIFAR10(
        root="~/data/",
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        download=True
    )
    test_dataset = datasets.CIFAR10(
        root="~/data/",
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        download=True
    )
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
        train_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        for step, (train_img, train_label) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            batch_count = step
            train_img = train_img.cuda(gpu, non_blocking=True)
            train_label = train_label.cuda(gpu, non_blocking=True)
            logits = model(train_img)
            loss = criterion(logits, train_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.rank == 0:
                train_loss += loss.item()
                _, predicted = logits.max(1)
                total += train_label.size(0)
                correct += predicted.eq(train_label).sum().item()
        
        if args.rank == 0:
            log["train_loss"].append(train_loss/(batch_count+1))
            log["train_acc"].append(100.*correct/total)
        
        if args.rank == 0:
            test_sampler.set_epoch(epoch)
            with torch.no_grad():
                test_loss = 0
                correct = 0
                total = 0
                batch_count = 0
                for step, (val_img, val_label) in tqdm(enumerate(test_loader)):
                    batch_count = step
                    val_img = val_img.cuda(gpu, non_blocking=True)
                    val_label = val_label.cuda(gpu, non_blocking=True)
                    logits = model(val_img)
                    loss = criterion(logits, val_label)
                
                    test_loss += loss.item()
                    _, predicted = logits.max(1)
                    total += val_label.size(0)
                    correct += predicted.eq(val_label).sum().item()
                
                log["test_loss"].append(test_loss/(batch_count+1))
                log["test_acc"].append(100.*correct/total)   
        
            print(f"Epoch: {epoch} - " + " - ".join([f"{key}: {log[key][epoch]}" for key in log]))
            
        scheduler.step()
    
    if args.rank == 0:
        log_df = pd.DataFrame(log)
        log_df.to_parquet(log_path)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Torch Multi-GPU Baseline',
                    description='This project conduct the benchmark of among batch sizes in multi-gpu',
                    epilog='ENJOY!!!')
    
    parser.add_argument('--bs', type = int, default=32,
                    help='batch size')
    parser.add_argument('--workers', type = int, default=4,
                    help='Number of processor used in data loader')
    parser.add_argument('--epochs', type = int, default=1,
                    help='# Epochs used in training')
    parser.add_argument('--lr', type=float, default=0.01, 
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--port', type=int, default=8080, help='Multi-GPU Training Port.')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    main(args=args)