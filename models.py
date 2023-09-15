import numpy as np
import pandas as pd
import random
import sys
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.io import read_image
import torchvision.transforms as T
from tqdm import tqdm

from isic_data_loaders import ISIC2020DataLoader


class Resnet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(Resnet50FeatureExtractor, self).__init__()
        self.rn_model = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.rn_model.children())[:-1])
        self.feature_dim = self.rn_model.fc.in_features # 2048
        # output_dimension = 4
        # self.fc = nn.Linear(self.rn_model.fc.in_features, output_dimension)
        # self.rn_model.fc = nn.Linear(self.rn_model.fc.in_features, output_dimension)
        
    def forward(self, x):
        # 32, 3, 224, 224
        x = self.features(x) # 32, 2048, 1, 1
        x = x.view(x.size(0), -1) # 32, 2048
        return x

class Resnet50Classifier(nn.Module):
    def __init__(self, output_dimension, num_experts: int, experts_dim: list, pretrained_extractor: str):
        super(Resnet50Classifier, self).__init__()
        self.num_experts = num_experts
        if pretrained_extractor:
            checkpoint = torch.load(pretrained_extractor)
            basenet = Resnet50Classifier(output_dimension=16, num_experts=1, experts_dim=[16], pretrained_extractor=None)
            basenet.load_state_dict(checkpoint['model_state_dict'])
            self.feature_extractor = basenet.feature_extractor
        else:
            self.feature_extractor = Resnet50FeatureExtractor() # shared feature extractor backbone
        
        # print("self.feature_extractor.feature_dim",self.feature_extractor.feature_dim) # 2048
        self.fc_experts = []
        for i in range(self.num_experts): # expert classifier heads
            self.fc_experts.append(nn.Linear(self.feature_extractor.feature_dim, experts_dim[i]))
        self.fc_experts = nn.ModuleList(self.fc_experts)
        
        self.gating = nn.Linear(np.sum(experts_dim), output_dimension)
    
    def train_one_expert(self, idx): # freeze all other experts
        for i in range(self.num_experts):
            if i != idx:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = False
            else:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # print("x.shape",x.shape) # [16, 3, 224, 224]
        x = self.feature_extractor(x) # 32, 2048
        experts_out = []
        for i in range(self.num_experts):
            experts_out.append(self.fc_experts[i](x)) # 32, 4
        x = torch.cat(experts_out, dim=1)
        x = self.gating(x) # 32, 16
        return x


class Resnet50Classifier2(nn.Module):
    def __init__(self, output_dimension, num_experts: int, experts_dim: list):
        super(Resnet50Classifier2, self).__init__()
        self.num_experts = num_experts
        self.experts_dim = experts_dim
        self.feature_extractor = Resnet50FeatureExtractor() # shared feature extractor backbone
        
        self.fc_experts = []
        for i in range(self.num_experts): # expert classifier heads: expert classes + "other"
            self.fc_experts.append(nn.Linear(self.feature_extractor.feature_dim, experts_dim[i]+1))
        self.fc_experts = nn.ModuleList(self.fc_experts)
        self.gating = nn.Linear(np.sum(experts_dim)+num_experts, output_dimension)
    
    def train_one_expert(self, idx): # freeze all other experts
        for i in range(self.num_experts):
            if i != idx:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = False
            else:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x) # 32, 2048
        experts_out = []
        for i in range(self.num_experts):
            experts_out.append(self.fc_experts[i](x)) # 32, 4
        x = torch.cat(experts_out, dim=1)
        x = self.gating(x) # 32, 16
        # x = self.aggregate_outputs(experts_out, 1)
        return x


class Resnet50Classifier3(nn.Module):
    def __init__(self, output_dimension, num_experts: int, experts_dim: list, pretrained_extractor: str = None): 
        super(Resnet50Classifier3, self).__init__()
        self.num_experts = num_experts
        self.experts_dim = experts_dim
        # if pretrained_extractor:
        #     checkpoint = torch.load(pretrained_extractor)
        #     basenet = Resnet50Classifier(output_dimension=16, num_experts=1, experts_dim=[16], pretrained_extractor=None)
        #     basenet.load_state_dict(checkpoint['model_state_dict'])
        #     self.feature_extractor = basenet.feature_extractor
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        # else:
        #     print("not pretrained extractor")
        self.feature_extractor = Resnet50FeatureExtractor() # shared feature extractor backbone
        
        self.fc_experts = []
        for i in range(self.num_experts): # expert classifier heads: expert classes + "other"
            self.fc_experts.append(nn.Linear(self.feature_extractor.feature_dim, experts_dim[i]+1))
        self.fc_experts = nn.ModuleList(self.fc_experts)
        # self.gating = nn.Linear(np.sum(experts_dim)+num_experts, output_dimension)
        
    def aggregate_outputs(self, experts_out, strategy):
        total_num_classes = np.sum(self.experts_dim)
        temp = np.cumsum(self.experts_dim)
        # expert_classes = {}
        # for i in range(self.num_experts):
        #     expert_classes[i] = list(range(temp[i]-self.experts_dim[i], temp[i]))
        # print(expert_classes)
        if strategy==2:
            sm = nn.Softmax(dim=1)
            for idx in range(self.num_experts):
                experts_out[idx] = sm(experts_out[idx])
        expert_idx = 0
        
        output_logit = []
        # for i in range(total_num_classes):
        for expert_idx in range(self.num_experts):
            # if temp[expert_idx] - 1 < i:
            #     expert_idx += 1
            # start = 0
            # if expert_idx >= 1:
            #     start = temp[expert_idx-1]
            # print("class i: ",i,"expert_idx: ", expert_idx)
            if strategy == 1: # multiply (not good, not using this)
                m = experts_out[expert_idx]
                for idx in range(self.num_experts):
                    if idx != expert_idx:
                        m *= torch.reshape(experts_out[idx][:,-1], (experts_out[idx][:,-1].shape[0], 1)).repeat(1,m.shape[1])
                output_logit.append(m)
            elif strategy == 2: # softmax then multiply the probabilities
                m = torch.clone(experts_out[expert_idx][:,:-1]) # 16,4
                for idx in range(self.num_experts):
                    if idx != expert_idx:
                        # m *= experts_out[idx][:,-1]
                        m *= torch.reshape(experts_out[idx][:,-1], (experts_out[idx][:,-1].shape[0], 1)).repeat(1,m.shape[1])
                output_logit.append(m) # output_probability
        x = torch.cat(output_logit, dim=1) # 16,16
        return x
    
    def convert_targets_expert(self, idx, targets): # convert targets to the label for this expert
        temp = np.cumsum(self.experts_dim)
        output = targets.clone().detach()
        
        start = 0
        if idx >= 1:
            start = temp[idx-1]

        for i in range(output.shape[0]):
            if not(output[i] >= start and output[i] < temp[idx]):
               output[i] = temp[idx]
        
        output -= start
        return output
    
    def train_one_expert(self, idx): # freeze all other experts
        for i in range(self.num_experts):
            if i != idx:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = False
            else:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x) # batch_size, 2048
        experts_out = []
        for i in range(self.num_experts):
            experts_out.append(self.fc_experts[i](x)) # batch_size, 5
        # x = torch.cat(experts_out, dim=1)
        # x = self.gating(x) # batch_size, 16
        aggregated_pred = self.aggregate_outputs(experts_out, 2)
        # return x
        return experts_out



class Resnet50Classifier4(nn.Module):
    def __init__(self, output_dimension, num_experts: int, experts_dim: list, pretrained_extractor: str):
        super(Resnet50Classifier4, self).__init__()
        self.num_experts = num_experts
        self.experts_dim = experts_dim
        # self.feature_extractor = Resnet50FeatureExtractor() # shared feature extractor backbone
        
        checkpoint = torch.load(pretrained_extractor)
        basenet = Resnet50Classifier(output_dimension=16, num_experts=1, experts_dim=[16], pretrained_extractor=None)
        basenet.load_state_dict(checkpoint['model_state_dict'])
        self.feature_extractor = basenet.feature_extractor
        # freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.fc_experts = []
        for i in range(self.num_experts): # expert classifier heads: expert classes + "other"
            self.fc_experts.append(nn.Linear(self.feature_extractor.feature_dim, experts_dim[i]+1))
        self.fc_experts = nn.ModuleList(self.fc_experts)
        # self.gating = nn.Linear(np.sum(experts_dim)+num_experts, output_dimension)
        
    def aggregate_outputs(self, experts_out, strategy):
        total_num_classes = np.sum(self.experts_dim)
        temp = np.cumsum(self.experts_dim)
        # expert_classes = {}
        # for i in range(self.num_experts):
        #     expert_classes[i] = list(range(temp[i]-self.experts_dim[i], temp[i]))
        # print(expert_classes)
        if strategy==2:
            sm = nn.Softmax(dim=1)
            for idx in range(self.num_experts):
                experts_out[idx] = sm(experts_out[idx])
        expert_idx = 0
        
        output_logit = []
        # for i in range(total_num_classes):
        for expert_idx in range(self.num_experts):
            # if temp[expert_idx] - 1 < i:
            #     expert_idx += 1
            # start = 0
            # if expert_idx >= 1:
            #     start = temp[expert_idx-1]
            # print("class i: ",i,"expert_idx: ", expert_idx)
            if strategy == 1: # multiply (not good, not using this)
                m = experts_out[expert_idx]
                for idx in range(self.num_experts):
                    if idx != expert_idx:
                        m *= torch.reshape(experts_out[idx][:,-1], (experts_out[idx][:,-1].shape[0], 1)).repeat(1,m.shape[1])
                output_logit.append(m)
            elif strategy == 2: # softmax then multiply the probabilities
                m = torch.clone(experts_out[expert_idx][:,:-1]) # 16,4
                for idx in range(self.num_experts):
                    if idx != expert_idx:
                        # m *= experts_out[idx][:,-1]
                        m *= torch.reshape(experts_out[idx][:,-1], (experts_out[idx][:,-1].shape[0], 1)).repeat(1,m.shape[1])
                output_logit.append(m) # output_probability
        x = torch.cat(output_logit, dim=1) # 16,16
        return x
    
    def convert_targets_expert(self, idx, targets): # convert targets to the label for this expert
        temp = np.cumsum(self.experts_dim)
        output = targets.clone().detach()
        
        start = 0
        if idx >= 1:
            start = temp[idx-1]

        for i in range(output.shape[0]):
            if not(output[i] >= start and output[i] < temp[idx]):
               output[i] = temp[idx]
        
        output -= start
        return output
    
    def train_one_expert(self, idx): # freeze all other experts
        for i in range(self.num_experts):
            if i != idx:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = False
            else:
                for param in self.fc_experts[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x) # batch_size, 2048
        experts_out = []
        for i in range(self.num_experts):
            experts_out.append(self.fc_experts[i](x)) # batch_size, 5
        # x = torch.cat(experts_out, dim=1)
        # x = self.gating(x) # batch_size, 16
        aggregated_pred = self.aggregate_outputs(experts_out, 2)
        # return x
        return experts_out



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_size = 16
    division_type=1
    num_experts=1
    dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts, dataset = "multiple2") # dataset = "orig", dataset = "multiple"
    train_dataloaders = dataloader.get_train_dataloaders()
    test_dataloader = dataloader.get_test_dataloader()
    iters = [iter(train_dataloaders[idx]) for idx in range(num_experts)]

    # testnet = Resnet50Classifier3(output_dimension=dataloader.total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num).to(device)
    testnet = Resnet50Classifier4(output_dimension=dataloader.total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num, pretrained_extractor="torch_model_checkpoints/2023_05_01_20_13_41/testnet_epoch37.pt").to(device)

    num_switches = 4
    experts_finished = [False] * num_experts
    # print(dataloader.experts_train_classes_size) # [28014, 2200, 589, 110]

    for idx in range(num_experts):
        idx = num_experts-1
        testnet.train_one_expert(idx)
        # print(train_dataloaders[idx].dataset.__len__(), len(train_dataloaders[idx]), dataloader.experts_train_classes_size[idx]) # 110 7 110
        # for i in range(dataloader.experts_train_classes_size[idx] // num_switches + 1):
        for i in range(len(train_dataloaders[idx])):
            data = next(iters[idx], None)
            if data==None:
                # print("here1: isNone")
                break
            images, labels = data[0].to(device), data[1].to(device)
            experts_out = testnet(images)
            print(len(experts_out), experts_out[0].shape)
            # print(testnet.convert_targets_expert(1,labels))
            break

        #     print(i),
        # for i in range(1000):
        #     data = next(iters[idx], None)
        #     print(i),
        #     if data==None:
        #         print("here2: isNone")
        #         experts_finished[idx] = True
        #         break
        break