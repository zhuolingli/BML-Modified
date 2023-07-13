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

class ImageNet(Dataset):
    def __init__(self, args, root_path=None, train_transform=None, test_transform=None, strong_train_trans=None,
                 fix_seed=True, split="train"):
        super(Dataset, self).__init__()
        self.fix_seed = fix_seed
        self.split = split
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.strong_train_trans = strong_train_trans

        self.dataset = args.dataset
        print("Current train dataset: ", self.dataset, self.split)
        if root_path is None:
            root_path = "data/few_shot/data"
        if self.dataset == "tieredImageNet":
            self.imgs = np.load(os.path.join(root_path, self.dataset, "{}_images.npz").format(self.split))["images"]
            self.origin_labels = \
                pickle.load(open(os.path.join(root_path, self.dataset, "{}_labels.pkl").format(self.split), "rb"),
                            encoding='latin1')["labels"]
            if self.split == "train":
                self.val_imgs = np.load(os.path.join(root_path, self.dataset, "val_images.npz"))["images"]
                self.val_labels = pickle.load(open(os.path.join(root_path, self.dataset, "val_labels.pkl"), "rb"),
                                              encoding='latin1')["labels"]
                self.imgs = np.concatenate([self.imgs, self.val_imgs], axis=0)
                self.origin_labels.extend(self.val_labels)
        elif self.dataset == "miniImageNet":
            if self.split == "test":
                split_name = "{}_category_split_test.pickle".format(self.dataset)
            elif self.split == "val":
                split_name = "{}_category_split_val.pickle".format(self.dataset)
            else:
                split_name = "{}_category_split_train_phase_train.pickle".format(self.dataset)
            with open(os.path.join(root_path, self.dataset, split_name), 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                self.imgs = data['data']
                self.origin_labels = data['labels']
        elif self.dataset == "FC100" or self.dataset == "CIFAR-FS":
            with open(os.path.join(root_path, self.dataset, "{}.pickle".format(self.split)), 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                self.imgs = data['data']
                self.origin_labels = data['labels']
        self.labels = self.origin_labels

        # img1 = self.imgs[1]
        # img1 = np.array(img1)
        # cv2.imwrite('img.jpg', img1)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data, label = self.imgs[item], self.labels[item]
        if self.split == "train":
            if self.strong_train_trans is not None:
                data = self.strong_train_trans.augment_image(data)
            else:
                data = data
            image = self.train_transform(data)
        else:
            image = self.test_transform(data)
        return image, label


def get_wnid2label_dict(split='train'):
    map_file = 'data/few_shot/data/miniImageNet/miniImagenet_Ravi/cls_'
    map_file = map_file + split + '.txt'
    
    wnid2gtlabel_dict = {}
    with open(map_file, 'r') as f:
        content = f.readlines()
        for idx, line in enumerate(content):
            wnid = line.split(' ')[1]
            wnid2gtlabel_dict.update({wnid:str(idx)})
    return wnid2gtlabel_dict

def get_dataloder(opt, split='train'):
    # transforms
    train_trans, test_trans, test_trans_plus = transforms_options[opt.transform] # 不同数据集，这里也不一样
    # # train
    trainset = ImageNet(args=opt, train_transform=test_trans, test_transform=test_trans, split=split)
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
    if config.dataset == 'miniImageNet':
        wnid2gtlabel_dict = get_wnid2label_dict(split)
        gtlabel2wnid_dict = {label:wnid for wnid,label in wnid2gtlabel_dict.items()}
            
        
    dataloader = get_dataloder(config, split)
    model = BMLBuilder(config)
    model.cuda()
    load_checkpoint(config, model)
    print('starting saveing!')

    model.eval()
        
    feats_list, img_metas_list, gt_labels_list = [], [], []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            # compute output
            img_metas_list.extend([gtlabel2wnid_dict[str(label)]  for label in labels.tolist()])
            output = model(images)
            feats = 0.5 * output[2] + 0.5 * output[3] # global + local
            feats_list.append(feats.cpu())
            gt_labels_list.append(labels)
    feats = torch.cat(feats_list, dim=0)
    gt_labels = torch.cat(gt_labels_list, dim=0)

    
    wnid2Embidx = {wnid:[] for wnid in wnid2gtlabel_dict.keys()}
    for idx, label in enumerate(gt_labels.tolist()):
        wnid = gtlabel2wnid_dict[str(label)]
        wnid2Embidx[wnid].append(idx)

    all_features = {}
    all_features['feats'] = feats
    all_features['gt_labels'] = gt_labels
    all_features['img_metas'] = img_metas_list
    all_features['wnid_to_gtlabel'] = wnid2gtlabel_dict
    with open(os.path.join(output_file, 'all_img_feats.pickle'), 'wb') as f:
        pkl.dump(all_features, f,protocol=pkl.HIGHEST_PROTOCOL)
    
    class_feature = {}
    for wnid, embidx in wnid2Embidx.items():
        emb = feats[embidx,:]
        mean = torch.mean(emb, dim=0).unsqueeze(dim=0)
        std = torch.std(emb, dim=0).unsqueeze(dim=0)
        class_feature[wnid] = {'mean':mean, 'std':std}
    with open(os.path.join(output_file, 'class_centroid.pickle'), 'wb') as f:
        pkl.dump(class_feature, f,protocol=pkl.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':    
    split = 'test'
    ckpt = 'params/miniImageNet/Res12_miniImageNet_ep100.pth'
    dataset = 'mini'
    FILE_TO_SAVE = '/run/media/cv/d/lzl/fsl/from_50/train/mmfewshot/my_output/BML2'

    
    output_file = os.path.join(FILE_TO_SAVE, dataset, split)
    if not osp.exists(output_file):
        os.makedirs(output_file)
        print('make dir:{}'.format(output_file))
    
    
    config = parse_option()
    config.ckp_path = ckpt
    config.dataset = 'miniImageNet' 
    config.transform = 'A'
    save_emb(config, output_file, split)
    # 这个py文件用来存取tired应该有问题，参见另一个