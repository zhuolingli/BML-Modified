import os
import torch
import pickle as pkl
from dataloader.transform.transform_cfg import transforms_options, transforms_list
from model.BML import BMLBuilder
import os.path as osp
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np
from dataloader.transform.transform_cfg import transforms_options
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
import cv2
from PIL import Image

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--mode", type=str, choices=["bml", "global", "local"], default="bml")
    parser.add_argument('--backbone', type=str, choices=["Res12", "Res18"], default="Res12")
    parser.add_argument('--eval_freq', type=int, default=20, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    # optimization
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--decay_step", type=int, default=40)
    parser.add_argument('--optim', type=str, choices=["adam", "SGD"], default="SGD")
    # dataset
    parser.add_argument('--dataset', type=str, default='CUB', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', "CUB"])
    parser.add_argument('--transform', type=str, default='E', choices=transforms_list)
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

class CUB(Dataset):
    def __init__(self, split, train_transform=None, test_transform=None):
        self.split = split
        self.wnids = []
        if self.split == "train":  # train with raw img
            self.root_path = "/run/media/cv/d/lzl/fsl/from_50/train/BML_mod/data/few_shot/data/CUB_200_2011"
            self.IMAGE_PATH = os.path.join(self.root_path, "images")
            self.SPLIT_PATH = os.path.join(self.root_path, "split")
        else:  # test with cropped img based on bounding box
            self.IMAGE_PATH = "/run/media/cv/d/lzl/fsl/from_50/train/BML_mod/data/few_shot/data/CUB_test"
            self.SPLIT_PATH = os.path.join(self.IMAGE_PATH, "split")
        txt_path = os.path.join(self.SPLIT_PATH, split + '.csv')
        self.data, self.labels = self.parse_csv(txt_path)
        self.num_class = np.unique(np.array(self.labels)).shape[0]
        print("Current {} dataset: CUB, {}_ways".format(split, self.num_class))
        self.train_transform = train_transform
        self.test_transform = test_transform

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            if self.split == "train":
                path = os.path.join(self.IMAGE_PATH, wnid, name)
            else:
                path = os.path.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.labels[i]
        if self.split == "train":
            image = self.train_transform(Image.open(data).convert('RGB'))
        else:
            image = self.test_transform(Image.open(data).convert('RGB'))
        return image, label



def get_dataloder(opt, split='train'):
    # transforms
    train_trans, test_trans, test_trans_plus = transforms_options[opt.transform] # 不同数据集，这里也不一样
    # # train
    trainset = CUB( train_transform=test_trans, test_transform=test_trans, split=split)
    meta_trainloader = DataLoader(dataset=trainset, batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)
    return meta_trainloader

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
        
    dataloader = get_dataloder(config, split)
    model = BMLBuilder(config)
    model.cuda()
    load_checkpoint(config, model)
    print('starting saveing!')

    model.eval()
        
    feats_list, img_metas_list, gt_labels_list = [], [], []
    
    gtlabel2wnid_dict = dataloader.dataset.wnids
    gtlabel2wnid_dict = {label:int(wnid.split('.')[0]) for label, wnid in enumerate(gtlabel2wnid_dict)}
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            # compute output
            if idx ==29:
                a= 1
                pass
            
            
            img_metas_list.extend([gtlabel2wnid_dict[label]  for label in labels.tolist()])
            output = model(images)
            feats = 0.5 * output[2] + 0.5 * output[3] # global + local
            feats_list.append(feats.cpu())
            gt_labels_list.append(labels)
    feats = torch.cat(feats_list, dim=0)
    gt_labels = torch.cat(gt_labels_list, dim=0)

    
    wnid2Embidx = {wnid:[] for wnid in  gtlabel2wnid_dict.values()}
    for idx, label in enumerate(gt_labels.tolist()):
        wnid = gtlabel2wnid_dict[label]
        wnid2Embidx[wnid].append(idx)

    all_features = {}
    all_features['feats'] = feats
    all_features['gt_labels'] = gt_labels
    all_features['img_metas'] = img_metas_list
    # all_features['wnid_to_gtlabel'] = wnid2gtlabel_dict
   
    # with open(os.path.join(output_file, 'all_img_feats.pickle'), 'wb') as f:
        # pkl.dump(all_features, f,protocol=pkl.HIGHEST_PROTOCOL)
    
    class_feature = {}
    for wnid, embidx in wnid2Embidx.items():
        emb = feats[embidx,:]
        mean = torch.mean(emb, dim=0).unsqueeze(dim=0)
        std = torch.std(emb, dim=0).unsqueeze(dim=0)
        class_feature[wnid] = {'mean':mean, 'std':std}
    # with open(os.path.join(output_file, 'class_centroid.pickle'), 'wb') as f:
        # pkl.dump(class_feature, f,protocol=pkl.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':    
    split = 'train'
    ckpt = 'params/CUB/Res12_CUB_ep100.pth'
    dataset = 'cub'
    FILE_TO_SAVE = '/run/media/cv/d/lzl/fsl/from_50/train/mmfewshot/my_output/BML'

    
    output_file = os.path.join(FILE_TO_SAVE, dataset, split)
    if not osp.exists(output_file):
        os.makedirs(output_file)
        print('make dir:{}'.format(output_file))
    
    
    config = parse_option()
    config.ckp_path = ckpt
    config.dataset = 'CUB' 
    config.transform = 'E'
    save_emb(config, output_file, split)