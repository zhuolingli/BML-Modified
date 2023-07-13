import os
import torch
import pickle as pkl
from tqdm import tqdm
from dataloader.transform.transform_cfg import transforms_options, transforms_list
from model.BML import BMLBuilder
from tiered_imagenet__MMF import TieredImageNetDataset
import os
import os.path as osp
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn, Tensor
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np
from dataloader.transform.transform_cfg import transforms_options
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
import cv2
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--mode", type=str, choices=["bml", "global", "local"], default="bml")
    parser.add_argument('--backbone', type=str, choices=["Res12", "Res18"], default="Res12")
    parser.add_argument('--eval_freq', type=int, default=20, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    # optimization
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--decay_step", type=int, default=40)
    parser.add_argument('--optim', type=str, choices=["adam", "SGD"], default="SGD")
    # dataset
    parser.add_argument('--dataset', type=str, default='tieredImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', "CUB"])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    # meta setting
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_train_ways', type=int, default=15, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_train_shots', type=int, default=5, metavar='N',
                        help='Number of shots during train')
    parser.add_argument('--n_train_queries', type=int, default=7, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--episodes', default=2000, help="test episode num")
    # bml setting
    parser.add_argument("--weights", type=str, default="2_1_0.5")
    parser.add_argument("--alpha_1", type=float, default=5.5)  # 2.5
    parser.add_argument("--alpha_2", type=float, default=0.1)
    parser.add_argument("--T", type=float, default=4.0)
    # spatial for single view training
    parser.add_argument("--spatial", action="store_true")
    # continue train
    parser.add_argument("--is_continue", action="store_true")
    # eval
    parser.add_argument("--is_eval", action="store_true")
    parser.add_argument("--ckp_path", type=str)
    
    # save setting
    parser.add_argument('-s', '--save_folder', type=str, default='params')
    parser.add_argument('--n_cls', type=int, default=64)
    parser.add_argument("--seed", default='0', type=str)
    
    opt = parser.parse_args()
    return opt



def load_checkpoint(config, model):
    model_dict = model.state_dict()
    checkpoint = torch.load(config.ckp_path)
    exist_pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
    model_dict.update(exist_pretrained_dict)
    msg = model.load_state_dict(model_dict, strict=False)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()

def save_emb(config,output_file, split):
    
    dataset = TieredImageNetDataset(split)
    dataloader = DataLoader(dataset=dataset,
                            num_workers=8,
                            batch_size=500,
                            )
    
    with open('data/few_shot/data/tieredImageNet/class_names.txt', 'r') as f:
        classnames= f.readlines()
    with open('data/few_shot/data/tieredImageNet/synsets.txt', 'r') as f:
        wnids= f.readlines()

    name2wnid_dict = {}
    count = 0
    for idx, (name, wnid) in enumerate(zip(classnames, wnids), 1):
        name = name.split('\n')[0].split(',')[0]
        wnid = wnid.split('\n')[0]
        # print(idx, name, wnid)
        count = count + 1
        if count != idx:
            print(idx, name, wnid)
        
        name2wnid_dict[name]= wnid
        
    model = BMLBuilder(config)
    model.cuda()
    load_checkpoint(config, model)
    model.eval()
    print('starting saveing!')
        
    feats_list, img_metas_list, gt_labels_list = [] ,[], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            if torch.cuda.is_available():
                data, label = [_.cuda() for _ in batch]
                output = model(data)
                output = output[2] + output[3]
                labels = label.cpu().tolist()
                gt_labels_list.extend(labels)
                feats_list.append(output.cpu())
        
   
    feats = torch.cat(feats_list, dim=0)
    gt_labels = torch.tensor(gt_labels_list)
    
    label2name_dict =  {label:name for name, label in dataloader.dataset.class_to_idx.items()}
    wnid2Embidx = {name2wnid_dict[name.split(',')[0]]:[] for name in dataloader.dataset.CLASSES}
    img_metas_list = [] # 重建metas信息，保存每个样本对应的wnid值
    for idx, label in enumerate(gt_labels_list):
        name =  label2name_dict[label].split(',')[0]  
        wnid = name2wnid_dict[name]
        img_metas_list.append(wnid)
        wnid2Embidx[wnid].append(idx)

    gt_labels = torch.tensor(gt_labels_list)

    all_features = {}
    all_features['feats'] = feats
    all_features['gt_labels'] = gt_labels
    all_features['img_metas'] = img_metas_list 
    
   
    class_feature = {}
    for wnid, embidx in wnid2Embidx.items():
        emb = feats[embidx,:]
        mean = torch.mean(emb, dim=0).unsqueeze(dim=0)
        std = torch.std(emb, dim=0).unsqueeze(dim=0)
        class_feature[wnid] = {'mean':mean, 'std':std}
    
    with open(os.path.join(output_file, 'all_img_feats.pickle'), 'wb') as f:
        pkl.dump(all_features, f,protocol=pkl.HIGHEST_PROTOCOL)
    
    
    with open(os.path.join(output_file, 'class_centroid.pickle'), 'wb') as f:
        pkl.dump(class_feature, f,protocol=pkl.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':    
    split = 'test'
    ckpt = 'params/tieredImageNet/Res12_tieredImageNet_ep180.pth'
    dataset = 'tiered'
    FILE_TO_SAVE = '/run/media/cv/d/lzl/fsl/from_50/train/mmfewshot/my_output/BML2'

    output_file = os.path.join(FILE_TO_SAVE, dataset, split)
    if not osp.exists(output_file):
        os.makedirs(output_file)
        print('make dir:{}'.format(output_file))
    
    
    config = parse_option()
    config.ckp_path = ckpt
    config.dataset = 'miniImageNet' if dataset=='mini' else 'tieredImageNet'
    config.transform = 'A' if dataset=='mini' else "B"
    with torch.no_grad():
        save_emb(config, output_file, split)
    