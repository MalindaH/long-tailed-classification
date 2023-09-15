import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image
from PIL import Image
import pandas as pd
import PIL
import json


isic_folder = '/home/lh9998/scratch/lh9998/images/'

metadata_df = pd.read_csv(isic_folder+'metadata.csv') # (71670, 26)
metadata_df = metadata_df[metadata_df['diagnosis'].notnull()] # 44232,26
metadata_df = metadata_df.sort_values('diagnosis')
metadata_df = metadata_df.reset_index(drop=True)

all_classes, class_counts = np.unique(metadata_df['diagnosis'], return_counts=True)
count_sort_ind = np.argsort(-class_counts) # sort from most to least
class_counts_sorted = -np.sort(-class_counts)
all_classes = all_classes[count_sort_ind]
num_classes = all_classes.shape[0]
# print(all_classes, class_counts_sorted)
major_classes = all_classes[np.where(class_counts_sorted > 10)]
other_idx = np.where(major_classes=='other')
major_classes = np.delete(major_classes, other_idx)
print(major_classes, len(major_classes)) # 16 classes
major_classes_sizes = np.delete(class_counts_sorted[np.where(class_counts_sorted > 10)], other_idx)
print(major_classes_sizes)


def get_expert_num_classes(num_classes, division_type = 1, num_experts = 4): # 0,1; 1,4
    # if division_type == 1: # divide classes sorted by class sizes evenly into num_experts
        expert_num_classes = []
        remainder = num_classes % num_experts
        quotient = num_classes // num_experts
        while remainder > 0:
            expert_num_classes.append(quotient+1)
            remainder -= 1
        while len(expert_num_classes) < num_experts:
            expert_num_classes.append(quotient)
        return expert_num_classes


def stratify_train_test_init(targets, num_classes):
    random.seed(123)
    
    class_inds = {}
    for idx in range(num_classes):
        class_inds[idx] = []
    for i in range(len(targets)): 
        class_inds[targets[i]].append(i)

    classes_train_inds = {}
    test_inds = set()
    # train_inds = set()
    for c in class_inds.keys():
        train_inds_temp = random.sample(class_inds[c], int(len(class_inds[c])*0.7))
        # train_inds.update(train_inds_temp)
        test_inds.update([item for item in class_inds[c] if item not in train_inds_temp])
        classes_train_inds[c] = list(train_inds_temp)
    # train_experts_inds = [list(train_inds)]
    test_inds = list(test_inds)
 
    return classes_train_inds, test_inds


def stratify_train_test(targets, num_classes, division_type = 0, num_experts = 1, multiply_dataset = 1): # 0,1; 1,4
    classes_train_inds, test_inds = stratify_train_test_init(targets, num_classes)
    
    class_inds = {}
    for idx in range(num_classes):
        class_inds[idx] = []
    for i in range(len(targets)): 
        class_inds[targets[i]].append(i)

    train_experts_inds = []
    # test_inds = set()
    if division_type == 0 and num_experts == 1: # get all samples
        train_inds = []
        for c in class_inds.keys():
            if multiply_dataset==1: # multiply minority classes by 20 times (average class size is 2000)
                if len(classes_train_inds[c]) < 200:
                    train_inds += classes_train_inds[c] * 20
                else:
                    train_inds += classes_train_inds[c]
            elif multiply_dataset==2:
                if len(classes_train_inds[c]) < 100:
                    train_inds += classes_train_inds[c] * 200
                elif len(classes_train_inds[c]) < 2000:
                    train_inds += classes_train_inds[c] * (2000 // len(classes_train_inds[c]))
                else:
                    train_inds += classes_train_inds[c]
            elif multiply_dataset==3:
                if len(classes_train_inds[c]) < 2000:
                    train_inds += classes_train_inds[c] * (2000 // len(classes_train_inds[c]))
                else:
                    train_inds += classes_train_inds[c]
            elif multiply_dataset==4: # "multiple4"
                baseline_accu = np.array([0.9467231897341888, 0.3896976483762598, 0.7982456140350878, 0.39019189765458423, 0.5454545454545454, 0.2724014336917563, 0.21739130434782608, 0.24031007751937986, 0.4, 0.6666666666666666, 0.014925373134328358, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
                class_tomultiply = np.round(np.reciprocal(baseline_accu)).astype(int)
                # print("class_tomultiply",np.reciprocal(baseline_accu), class_tomultiply) 
                # [  1.05627496   2.56609195   1.25274725   2.56284153   1.83333333 3.67105263   4.6          4.16129032   2.5          1.5 67.         100.         100.         100.         100. 100.        ] [  1   3   1   3   2   4   5   4   2   2  67 100 100 100 100 100]
                train_inds += classes_train_inds[c] * class_tomultiply[c]
            elif multiply_dataset==5: # "multiple5"
                baseline_accu = np.array([0.9467231897341888, 0.3896976483762598, 0.7982456140350878, 0.39019189765458423, 0.5454545454545454, 0.2724014336917563, 0.21739130434782608, 0.24031007751937986, 0.4, 0.6666666666666666, 0.014925373134328358, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
                class_tomultiply = np.round(np.reciprocal(baseline_accu)).astype(int)
                # print("class_tomultiply",np.reciprocal(baseline_accu), class_tomultiply) 
                # [  1.05627496   2.56609195   1.25274725   2.56284153   1.83333333 3.67105263   4.6    4.16129032   2.5     1.5 67.   10000.    10000.    10000.    10000.    10000.] [  1   3   1   3   2   4   5   4   2   2  67 10000 10000 10000 10000 10000]
                train_inds += classes_train_inds[c] * class_tomultiply[c]
            elif multiply_dataset==21:
                if len(classes_train_inds[c]) > 1000: # make all class sizes 1000
                    train_inds += list(random.sample(classes_train_inds[c], 1000))
                else:
                    train_inds += classes_train_inds[c] * (1000 // len(classes_train_inds[c]))
            else:
                train_inds += classes_train_inds[c]
            # train_inds.update(classes_train_inds[c])
            # test_inds.update([item for item in class_inds[c] if item not in train_inds_temp])              
        # train_experts_inds = [list(train_inds)]
        train_experts_inds = [train_inds]

    elif division_type == 1: # divide classes sorted by class sizes evenly into num_experts
        ## now need to handle multiple1, multiple2, multiple3 with multiple experts
        expert_num_classes = get_expert_num_classes(num_classes, division_type, num_experts)
        c = 0
        for e in range(num_experts):
            train_inds = []
            # train_inds = set()
            for _ in range(expert_num_classes[e]):
                if multiply_dataset==1: # multiply minority classes by 20 times (average class size is 2000)
                    if len(classes_train_inds[c]) < 200:
                        train_inds += classes_train_inds[c] * 20
                    else:
                        train_inds += classes_train_inds[c]
                elif multiply_dataset==2:
                    if len(classes_train_inds[c]) < 100:
                        train_inds += classes_train_inds[c] * 200
                    elif len(classes_train_inds[c]) < 2000:
                        train_inds += classes_train_inds[c] * (2000 // len(classes_train_inds[c]))
                    else:
                        train_inds += classes_train_inds[c]
                elif multiply_dataset==3:
                    if len(classes_train_inds[c]) < 2000:
                        train_inds += classes_train_inds[c] * (2000 // len(classes_train_inds[c]))
                    else:
                        train_inds += classes_train_inds[c]
                # train_inds.update(classes_train_inds[c])
                elif multiply_dataset==4: # "multiple4"
                    baseline_accu = np.array([0.9467231897341888, 0.3896976483762598, 0.7982456140350878, 0.39019189765458423, 0.5454545454545454, 0.2724014336917563, 0.21739130434782608, 0.24031007751937986, 0.4, 0.6666666666666666, 0.014925373134328358, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
                    class_tomultiply = np.round(np.reciprocal(baseline_accu)).astype(int)
                    train_inds += classes_train_inds[c] * class_tomultiply[c] # becomes 75290 training samples in total
                elif multiply_dataset==5: # "multiple5"
                    baseline_accu = np.array([0.9467231897341888, 0.3896976483762598, 0.7982456140350878, 0.39019189765458423, 0.5454545454545454, 0.2724014336917563, 0.21739130434782608, 0.24031007751937986, 0.4, 0.6666666666666666, 0.014925373134328358, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
                    class_tomultiply = np.round(np.reciprocal(baseline_accu)).astype(int)
                    train_inds += classes_train_inds[c] * class_tomultiply[c]
                # elif multiply_dataset==21:
                #     if len(classes_train_inds[c]) > 1000: # make all class sizes 1000
                #         train_inds += list(random.sample(classes_train_inds[c], 1000))
                #     else:
                #         train_inds += classes_train_inds[c] * (1000 // len(classes_train_inds[c]))
                else:
                    train_inds += classes_train_inds[c]
                c += 1
            train_experts_inds.append(train_inds)
        # test_inds = list(test_inds)
    # elif division_type == 2: # one expert with all classes, similar number of samples from each class; second expert with fewer classes, ...
    return train_experts_inds, test_inds




class ISIC2020Dataset(Dataset):
    def __init__(self, isic_folder, metadata_df, train, transform=None, division_type=0, num_experts=1, expert_index = 0):
        self.isic_folder = isic_folder
        self.metadata_df = metadata_df

        self.metadata_df = self.metadata_df.loc[[a in major_classes for a in self.metadata_df['diagnosis']], :]
        self.metadata_df = self.metadata_df.reset_index(drop=True)
        
        self.img_names = self.metadata_df['isic_id']
        self.transform = transform
        
        # convert str labels to int
        self.metadata_df['labels'] = self.metadata_df['diagnosis'].replace(major_classes,list(range(len(major_classes))))
        self.targets = self.metadata_df['labels']
        
        # # to generate dataset zip for styleGAN:
        # data = []
        # for idx in range(len(self.targets)):
        #     data.append([self.img_names[idx]+'.JPG', int(self.targets[idx])])
        # jsondata = {"labels": data}
        # print(jsondata)
        # with open('/home/lh9998/scratch/lh9998/images/dataset.json', 'w') as f:
        #     json.dump(jsondata, f)
        
        train_experts_inds, test_inds = stratify_train_test(self.targets, len(major_classes), division_type=division_type, num_experts=num_experts)
        # print(train_experts_inds[expert_index], expert_index)
        train_inds = train_experts_inds[expert_index]
        if train: # training dataset
            self.targets = self.targets[train_inds]
            self.img_names = self.img_names[train_inds]
            self.targets = self.targets.reset_index(drop=True)
            self.img_names = self.img_names.reset_index(drop=True)
        else: # testing dataset
            self.targets = self.targets[test_inds]
            self.img_names = self.img_names[test_inds]
            self.targets = self.targets.reset_index(drop=True)
            self.img_names = self.img_names.reset_index(drop=True)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.isic_folder+self.img_names[idx]+'.JPG', mode='r')
        # print("image.shape", image.size) # > 600,450, lots 1024
        # image = read_image(self.isic_folder+self.img_names[idx]+'.JPG')
        if self.transform:
            image = self.transform(image) # 3,224,224
        label = self.targets[idx]
        return image, label

# all_data_major = ISIC2020Dataset(isic_folder, metadata_df, True, None)
# print(all_data_major.targets)




class ISIC2020DatasetMultiple(Dataset): ## now need to handle multiple experts
    def __init__(self, isic_folder, metadata_df, train=True, transform=None, multiply_method=1, division_type=0, num_experts=1, expert_index = 0):
        self.isic_folder = isic_folder
        self.metadata_df = metadata_df

        self.metadata_df = self.metadata_df.loc[[a in major_classes for a in self.metadata_df['diagnosis']], :]
        self.metadata_df = self.metadata_df.reset_index(drop=True)
        
        self.img_names = self.metadata_df['isic_id']
        self.transform = transform
        
        # convert str labels to int
        self.metadata_df['labels'] = self.metadata_df['diagnosis'].replace(major_classes,list(range(len(major_classes))))
        self.targets = self.metadata_df['labels']
        
        # train_experts_inds, self.test_inds = stratify_train_test(self.targets, len(major_classes), division_type=0, multiply_dataset = multiply_method) # train 30913 -> 44194; test 13259
        # # print(train_experts_inds[expert_index], expert_index)
        # self.train_inds = train_experts_inds[0]
        
        train_experts_inds, test_inds = stratify_train_test(self.targets, len(major_classes), division_type=division_type, num_experts=num_experts, multiply_dataset = multiply_method)
        train_inds = train_experts_inds[expert_index]
        if train: # training dataset
            self.targets = self.targets[train_inds]
            self.img_names = self.img_names[train_inds]
            self.targets = self.targets.reset_index(drop=True)
            self.img_names = self.img_names.reset_index(drop=True)
        else: # testing dataset
            self.targets = self.targets[test_inds]
            self.img_names = self.img_names[test_inds]
            self.targets = self.targets.reset_index(drop=True)
            self.img_names = self.img_names.reset_index(drop=True)
        
        # if train: # training dataset
        #     self.targets = self.targets[self.train_inds]
        #     self.img_names = self.img_names[self.train_inds]
        #     self.targets = self.targets.reset_index(drop=True)
        #     self.img_names = self.img_names.reset_index(drop=True)
        # else: # testing dataset
        #     self.targets = self.targets[self.test_inds]
        #     self.img_names = self.img_names[self.test_inds]
        #     self.targets = self.targets.reset_index(drop=True)
        #     self.img_names = self.img_names.reset_index(drop=True)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.isic_folder+self.img_names[idx]+'.JPG', mode='r')
        # print("image.shape", image.size) # > 600,450, lots 1024
        # image = read_image(self.isic_folder+self.img_names[idx]+'.JPG')
        if self.transform:
            image = self.transform(image) # 3,224,224
        label = self.targets[idx]
        return image, label
    


class ISIC2020DataLoader(DataLoader):
    """
    Load ISIC 2020
    """
    def __init__(self, batch_size, shuffle=True, num_workers=4, division_type=0, num_experts=1, dataset="orig"):
        assert (dataset=="orig" or dataset.startswith("multiple") or dataset.startswith("balance"))
        
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        train_trsfm = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            # transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0,180)),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        self.division_type = division_type
        self.num_experts = num_experts
        self.total_num_classes = len(major_classes)
        self.total_classes_sizes = major_classes_sizes
        
        self.train_datasets = []
        self.experts_train_classes = []
        self.experts_train_classes_num = []
        self.experts_train_classes_size = []
        if dataset=="orig":
            print("origgggg dataset", num_experts)
            for i in range(num_experts):
                self.train_datasets.append(ISIC2020Dataset(isic_folder, metadata_df, train=True, transform=train_trsfm, division_type=division_type, num_experts=num_experts, expert_index=i))
                self.experts_train_classes.append(np.unique(self.train_datasets[i].targets))
                self.experts_train_classes_num.append(len(self.experts_train_classes[i]))
                self.experts_train_classes_size.append(len(self.train_datasets[i].targets))
            self.val_dataset = ISIC2020Dataset(isic_folder, metadata_df, train=False, transform=test_trsfm, division_type=division_type, num_experts=num_experts)
        # elif dataset == "multiple1" or dataset == "multiple2" or dataset == "multiple3": # only one expert
        elif dataset.startswith("multiple"): # multiple1, multiple2, multiple3 ## now need to handle multiple experts
            multiply_method = int(dataset[-1:])
            print("dataset", dataset, multiply_method)
            for i in range(num_experts):
                self.train_datasets.append(ISIC2020DatasetMultiple(isic_folder, metadata_df, train=True, transform=train_trsfm, multiply_method=multiply_method, division_type=division_type, num_experts=num_experts, expert_index=i))
                self.experts_train_classes.append(np.unique(self.train_datasets[i].targets))
                self.experts_train_classes_num.append(len(self.experts_train_classes[i]))
                self.experts_train_classes_size.append(len(self.train_datasets[i].targets))
            self.val_dataset = ISIC2020DatasetMultiple(isic_folder, metadata_df, train=False, transform=test_trsfm, multiply_method=multiply_method)
        elif dataset.startswith("balance"): # only one expert, balance1
            multiply_method = 20+int(dataset[-1:])
            self.train_datasets.append(ISIC2020DatasetMultiple(isic_folder, metadata_df, train=True, transform=train_trsfm, multiply_method=multiply_method))
            self.experts_train_classes.append(np.unique(self.train_datasets[0].targets))
            self.experts_train_classes_num.append(len(self.experts_train_classes[0]))
            self.experts_train_classes_size.append(len(self.train_datasets[0].targets))
            self.val_dataset = ISIC2020DatasetMultiple(isic_folder, metadata_df, train=False, transform=test_trsfm, multiply_method=multiply_method)
        
        # print(self.experts_train_classes, self.experts_train_classes_size)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        
    def get_train_dataloaders(self):
        ret = []
        for idx in range(self.num_experts):
            ret.append(self.get_train_dataloader(idx))
        return ret
    
    def get_train_dataloader(self, idx):
        print(idx, len(self.train_datasets[idx]))
        return DataLoader(dataset=self.train_datasets[idx], **self.init_kwargs)
    
    def get_test_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)



if __name__ == '__main__':
    # dataloader = ISIC2020DataLoader(16, division_type=1, num_experts=16, dataset="orig")
    dataloader = ISIC2020DataLoader(16, division_type=1, num_experts=16, dataset="multiple3")
    train_dataloaders = dataloader.get_train_dataloaders()
    test_dataloader = dataloader.get_test_dataloader()

    for data, target in train_dataloaders[1]:
        print(data.shape, target.shape)
        break