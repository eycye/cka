import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import os
from collections import OrderedDict
from torch_cka import CKA
import resnet_widths_all
import argparse
import json
import resnet_width_4
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--checkpoint', default='/home/eecs/eliciaye/ww_train/res18-25_2', help='saved model directory')
parser.add_argument('--method', default='', type=str, help='use ww or baseline')
parser.add_argument('--width_frac', default=1,type=float, help='fraction of original width')
parser.add_argument('--depth', default=18, type=int, help='ResNet depth')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

batch_size = 100 # eval batch size

dataset = CIFAR100(root='../data1/',
                  train=False,
                  download=False,
                  transform=transform)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)

model1 = resnet_width_4.ResNet18()

# luigi /data/eliciaye/val_experiments/me-prune/cifar100/resnet34-0.2/ckpt.pth'
# ckpt = torch.load('')
ckpt=torch.load(f'{args.checkpoint}/ckpt.pth')
state_dict = ckpt['net']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v

model1.load_state_dict(state_dict,strict=False)

model1 = model1.to(device)
# model1 = torch.nn.DataParallel(model1)
# cudnn.benchmark = True

cka = CKA(model1, model1,
        model1_name=f"ResNet{args.depth}-{args.width_frac*100}%", model2_name=f"ResNet{args.depth}-{args.width_frac*100}%, init_lr={args.lr}",
        device=device)

cka.compare(dataloader)

cka.plot_results(title=f"CKA Similarity between Model Layers ({args.method})",save_path=f"/home/eecs/eliciaye/cka/cka_images/resnet{args.depth}_{int(args.width_frac*100)}_{args.method}{args.lr}_trythis1.png")