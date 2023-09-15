import argparse
import numpy as np
import pandas as pd
import random
import sys
import os
import json
import torch
from torch import nn
import torch.nn.functional as f
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
import torchvision.transforms as T
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from isic_data_loaders import ISIC2020DataLoader, get_expert_num_classes
from models import Resnet50Classifier, Resnet50Classifier2, Resnet50Classifier3, Resnet50Classifier4
from metric import accuracy, class_avg_accuracy, class_avg_accuracy_misclassifymat
from focalloss import focal_loss




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train0():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    def eval_test(model, val_dataloader, num_classes, f):
        print("Evaluate on validation set...")
        f.write("Evaluate on validation set...\n")
        model.eval()

        agg_accu = []
        agg_avg_accu = []
        agg_class_accu = []
        for i, data in enumerate(val_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images) 
            
            accu = accuracy(outputs, labels)
            avg_accu, class_accu = class_avg_accuracy(outputs, labels)
            # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            
            agg_accu.append(accu)
            agg_avg_accu.append(avg_accu)
            agg_class_accu.append(class_accu)

        print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
        print("Validation average class accuracy: ", sum(agg_avg_accu)/len(agg_avg_accu))
        print("Validation class accuracies: ", np.mean(np.array(agg_class_accu), axis=0))
        f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
        f.write(f'Validation average class accuracy: {sum(agg_avg_accu)/len(agg_avg_accu)}\n')
        f.write(f'Validation class accuracies: {np.mean(np.array(agg_class_accu), axis=0)}\n')
        
        model.train()
        return agg_accu, agg_avg_accu, agg_class_accu


    epochs = 50

    batch_size = 32
    division_type=1
    num_experts=4

    model_path = "torch_model_checkpoints/"
    model_folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"/"
    log_path = model_path+model_folder_name
    os.mkdir(log_path)

    f = open(log_path+"config.log", "a")
    f.write(f'epochs: {epochs}\n')
    f.write(f'batch_size: {batch_size}\n')
    f.write(f'division_type: {division_type}\n')
    f.write(f'num_experts: {num_experts}\n')
    f.write("\n")


    dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts)
    train_dataloaders = dataloader.get_train_dataloaders()
    test_dataloader = dataloader.get_test_dataloader()
    total_num_classes = dataloader.total_num_classes
    total_classes_sizes = dataloader.total_classes_sizes
    # print(dataloader.experts_train_classes_size) # [28014, 2200, 589, 110]

    testnet = Resnet50Classifier(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_size).to(device)
    # testnet.feature_extractor = testnet.feature_extractor.to(device)

    # # weighted by class frequency
    # class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=float))
    # criterion = nn.CrossEntropyLoss(weight = class_freq_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(testnet.parameters(), lr=0.01)

    for epoch in tqdm(range(epochs)):
        f.write(f'Epoch: {epoch}\n')
        num_switches = 4
        experts_finished = [False] * num_experts
        iters = [iter(train_dataloaders[idx]) for idx in range(num_experts)]

        idx = 0
        while not all(experts_finished):
            print("expert idx:",idx)
            f.write(f'expert idx: {idx}\n')
            if experts_finished[idx]:
                idx = (idx+1)%num_experts
                break
            
            training_losses = []
            testnet.train_one_expert(idx)
            for i in range(len(train_dataloaders[idx]) // num_switches + 1):
                data = next(iters[idx], None)
                if data==None:
                    experts_finished[idx] = True
                    print("experts_finished, idx=", idx)
                    f.write('experts_finished, idx = {idx}\n')
                    idx = (idx+random.randint(1,num_experts-1))%num_experts
                    break
                
                images, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = testnet(images)
                loss = criterion(outputs, labels)
                
                training_losses.append(loss.item())

                loss.backward()
                optimizer.step()
            
            if len(training_losses) > 0:
                print("Average training loss:", sum(training_losses)/len(training_losses))
                f.write(f'Average training loss: {sum(training_losses)/len(training_losses)}\n')
            
                agg_accu, agg_avg_accu, agg_class_accu = eval_test(testnet, test_dataloader, total_num_classes, f)

        torch.save(testnet.state_dict(), f'{log_path}testnet_epoch{epoch}')
        print(f'Model saved as: {log_path}testnet_epoch{epoch}')
        f.write(f'Model saved as: {log_path}testnet_epoch{epoch}\n')
        f.write("\n")

    f.close()






#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_separate_expert_dataloaders(config):
    assert not "model" in config or config["model"] == "Resnet50Classifier" or config["model"] == "Resnet50Classifier2"
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # def eval_test(model, val_dataloader, num_classes, f):
    #     print("Evaluate on validation set...")
    #     if not config["no_write"]:
    #         f.write("Evaluate on validation set...\n")
    #     model.eval()

    #     agg_accu = []
    #     agg_avg_accu = []
    #     agg_class_accu = []
    #     # print("len(val_dataloader)",len(val_dataloader)) # 829
    #     for i, data in enumerate(val_dataloader):
    #         images, labels = data[0].to(device), data[1].to(device)

    #         outputs = model(images) 
            
    #         accu = accuracy(outputs, labels)
    #         avg_accu, class_accu = class_avg_accuracy(outputs, labels)
    #         # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            
    #         agg_accu.append(accu)
    #         agg_avg_accu.append(avg_accu)
    #         agg_class_accu.append(class_accu)

    #     print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
    #     print("Validation average class accuracy: ", sum(agg_avg_accu)/len(agg_avg_accu))
    #     print("Validation class accuracies: ", np.mean(np.array(agg_class_accu), axis=0))
    #     if not config["no_write"]:
    #         f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
    #         f.write(f'Validation average class accuracy: {sum(agg_avg_accu)/len(agg_avg_accu)}\n')
    #         f.write(f'Validation class accuracies: {np.mean(np.array(agg_class_accu), axis=0)}\n')
        
    #     model.train()
    #     return agg_accu, agg_avg_accu, agg_class_accu
    def eval_test(model, val_dataloader, num_classes, f, log_path=None, epoch=999):
        print("Evaluate on validation set...")
        if not config["no_write"]:
            f.write("Evaluate on validation set...\n")
        model.eval()

        agg_accu = []
        agg_corrects = np.zeros(num_classes, dtype=int)
        agg_lengths = np.zeros(num_classes, dtype=int)
        misclassify_mats = torch.zeros((num_classes, num_classes), dtype=int)
        # print("len(val_dataloader)",len(val_dataloader)) # 829
        for i, data in enumerate(val_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images) 
            
            accu = accuracy(outputs, labels)
            # avg_accu, class_accu = class_avg_accuracy(outputs, labels)
            # avg_accu, class_accu, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            corrects, lengths, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            # print("corrects, lengths:",corrects, lengths)
            
            agg_accu.append(accu)
            # agg_class_accu.append(class_accu)
            agg_corrects += corrects
            agg_lengths += lengths
            misclassify_mats += misclassify_mat

        print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
        print("Validation # corrects per class and class sizes: ", agg_corrects, agg_lengths)
        agg_class_accu = [(agg_corrects[c] / agg_lengths[c]) for c in range(num_classes)] # if lengths[c] > 0 else None
        print("Validation class accuracies:", agg_class_accu)
        print("Validation average class accuracy: ", np.mean(np.array(agg_class_accu), axis=0))
        # print("Validation misclassify_mats: ",misclassify_mats)
        misclassify_mats_normalized = misclassify_mats / misclassify_mats.sum(dim=-1).unsqueeze(-1)
        # print("Validation misclassify_mats_normalized: ", misclassify_mats_normalized)
        if not config["no_write"]:
            f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
            f.write(f'Validation # corrects per class and class sizes: {agg_corrects}, {agg_lengths}\n')
            f.write(f'Validation class accuracies: {agg_class_accu}\n')
            f.write(f'Validation average class accuracy: {np.mean(np.array(agg_class_accu), axis=0)}\n')
            f.write(f'Validation misclassify_mats: {misclassify_mats}\n')
            f.write(f'Validation misclassify_mats_normalized: {misclassify_mats_normalized}\n')
        
        if log_path is not None:
            plt.imshow(misclassify_mats_normalized, cmap="Oranges", interpolation='nearest')
            plt.colorbar()
            plt.savefig(log_path+"misclassify_mats_normalized_"+str(epoch)+".png")
            plt.clf()
    
        model.train()


    epochs = config["epochs"]

    batch_size = config["batch_size"]
    division_type=1
    num_experts=config["num_experts"]
    lr = config["lr"]

    if not config["no_write"]:
        model_path = "torch_model_checkpoints/"
        model_folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"/"
        log_path = model_path+model_folder_name
        os.mkdir(log_path)
    
        print("Write to log file: "+log_path+"config.log")
        f = open(log_path+"config.log", "a")
        config_file = config["config_file"]
        f.write(f'Read training config from file: {config_file}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'division_type: {division_type}\n')
        f.write(f'num_experts: {num_experts}\n')
        f.write(f'lr: {lr}\n')
        f.write("\n")
    else:
        f = None

    if "datasetmultiple" in config:
        dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts, dataset = config["datasetmultiple"]) # dataset = "orig", dataset = "multiple"
    else:
        dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts)
    train_dataloaders = dataloader.get_train_dataloaders()
    test_dataloader = dataloader.get_test_dataloader()
    total_num_classes = dataloader.total_num_classes
    total_classes_sizes = dataloader.total_classes_sizes
    # print(dataloader.experts_train_classes_size) # [28014, 2200, 589, 110]

    if not "model" in config or config["model"] == "Resnet50Classifier":
        if "pretrained_extractor" in config:
            testnet = Resnet50Classifier(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num, pretrained_extractor=config["pretrained_extractor"]).to(device)
        else:
            testnet = Resnet50Classifier(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num, pretrained_extractor=None).to(device)
    elif config["model"] == "Resnet50Classifier2":
        testnet = Resnet50Classifier2(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num).to(device)
    # elif config["model"] == "Resnet50Classifier3":
    #     testnet = Resnet50Classifier3(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num).to(device)

    if config["loss"]["type"] == "focal_loss" and not config["loss"]["weighted"]:
        # criterion = torch.hub.load(
        #     'adeelh/pytorch-multi-class-focal-loss',
        #     model='FocalLoss',
        #     alpha=torch.tensor([1/total_num_classes]*total_num_classes).to(device),
        #     # alpha=class_freq_weights,
        #     gamma=2,
        #     reduction='mean',
        #     force_reload=False
        # )
        criterion = focal_loss(alpha = torch.tensor([1/total_num_classes]*total_num_classes).to(device),
               gamma = 2,
               reduction = 'mean',
               device=device)
    elif config["loss"]["type"] == "focal_loss" and config["loss"]["weighted"]:
        class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=np.float32))
        class_freq_weights /= class_freq_weights.sum()
        class_freq_weights = class_freq_weights.to(device)
        # criterion = torch.hub.load(
        #     'adeelh/pytorch-multi-class-focal-loss',
        #     model='FocalLoss',
        #     # alpha=torch.tensor([1/total_num_classes]*total_num_classes).to(device),
        #     alpha=class_freq_weights,
        #     gamma=2,
        #     reduction='mean',
        #     force_reload=False
        # )
        criterion = focal_loss(alpha = class_freq_weights,
               gamma = 2,
               reduction = 'mean',
               device=device)
    elif config["loss"]["type"] == "CELoss" and config["loss"]["weighted"]:
        # CELoss weighted by class frequency
        class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=np.float32) * 100).to(device)
        # print(class_freq_weights) # [3.4375e-03, 1.6798e-02, 2.9265e-02, 6.4020e-02, 9.0909e-02, 1.0764e-01,
        # 1.4535e-01, 2.3364e-01, 3.7879e-01, 3.8610e-01, 4.5045e-01, 1.0101e+00,
        # 1.0989e+00, 3.2258e+00, 4.5455e+00, 6.2500e+00]
        criterion = nn.CrossEntropyLoss(weight = class_freq_weights)
    elif config["loss"]["type"] == "CELoss" and not config["loss"]["weighted"]:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(testnet.parameters(), lr=lr)

    if config["load_from_checkpoint"]:
        path = config["load_from_checkpoint"]
        checkpoint = torch.load(path)
        testnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print(f'Loaded model from checkpoint {path}')
        if not config["no_write"]:
            f.write(f'Loaded model from checkpoint {path}\n')
    
    if config["eval"]:
        testnet.eval()
        eval_test(testnet, test_dataloader, total_num_classes, f, log_path)
    else:
        testnet.train()
        for epoch in tqdm(range(epochs)):
            print(f'Epoch: {epoch}')
            if not config["no_write"]:
                f = open(log_path+"config.log", "a")
                f.write(f'Epoch: {epoch}\n')
            num_switches = 4
            experts_finished = [False] * num_experts
            iters = [iter(train_dataloaders[idx]) for idx in range(num_experts)]

            idx = 0
            while not all(experts_finished):
                print("\nexpert idx:",idx)
                if not config["no_write"]:
                    f.write(f'\nexpert idx: {idx}\n')
                if experts_finished[idx]:
                    idx = (idx+1)%num_experts
                    break
                
                training_losses = []
                testnet.train_one_expert(idx)
                for i in range(len(train_dataloaders[idx]) // num_switches + 1):
                    data = next(iters[idx], None)
                    if data==None:
                        experts_finished[idx] = True
                        print("experts_finished, idx=", idx)
                        if not config["no_write"]:
                            f.write(f'experts_finished, idx = {idx}\n')
                        if num_experts > 1:
                            idx = (idx+random.randint(1,num_experts-1))%num_experts
                        break
                    
                    images, labels = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = testnet(images)
                    
                    loss = criterion(outputs, labels)
                    # print(loss.item())
                    
                    training_losses.append(loss.item())

                    loss.backward()
                    optimizer.step()
                
                if len(training_losses) > 0:
                    avg_train_loss = sum(training_losses)/len(training_losses)
                    print("Average training loss:", avg_train_loss)
                    if not config["no_write"]:
                        f.write(f'Average training loss: {avg_train_loss}\n')
                
                    if not config["no_write"]:
                        eval_test(testnet, test_dataloader, total_num_classes, f, log_path, epoch=epoch)
                    else:
                        eval_test(testnet, test_dataloader, total_num_classes, f, None, epoch=epoch)

            if not config["no_write"] and epoch%1 == 0:
                torch.save({
                    'model_state_dict': testnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'{log_path}testnet_epoch{epoch}.pt')
                print(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt')
                f.write(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt\n')
            if not config["no_write"]:
                f.write("\n")
                f.close()






#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_separate_expert_dataloaders_classifier3(config):
    assert config["model"] == "Resnet50Classifier3"
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # def eval_test(model, val_dataloader, num_classes, f):
    #     print("Evaluate on validation set...")
    #     if not config["no_write"]:
    #         f.write("Evaluate on validation set...\n")
    #     model.eval()

    #     agg_accu = []
    #     agg_avg_accu = []
    #     agg_class_accu = []
    #     # print("len(val_dataloader)",len(val_dataloader)) # 829
    #     for i, data in enumerate(val_dataloader):
    #         images, labels = data[0].to(device), data[1].to(device)

    #         outputs = model(images) 
    #         outputs = testnet.aggregate_outputs(outputs, 2) # [16, 16]
    #         # print("outputs, labels:",outputs.shape, outputs, labels) 
    #         # print("torch.argmax(output, dim=1):", torch.argmax(outputs, dim=1))
    #         accu = accuracy(outputs, labels)
    #         avg_accu, class_accu = class_avg_accuracy(outputs, labels)
    #         # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            
    #         agg_accu.append(accu)
    #         agg_avg_accu.append(avg_accu)
    #         agg_class_accu.append(class_accu)

    #     print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
    #     print("Validation average class accuracy: ", sum(agg_avg_accu)/len(agg_avg_accu))
    #     print("Validation class accuracies: ", np.mean(np.array(agg_class_accu), axis=0))
    #     if not config["no_write"]:
    #         f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
    #         f.write(f'Validation average class accuracy: {sum(agg_avg_accu)/len(agg_avg_accu)}\n')
    #         f.write(f'Validation class accuracies: {np.mean(np.array(agg_class_accu), axis=0)}\n')
        
    #     model.train()
    #     return agg_accu, agg_avg_accu, agg_class_accu

    def eval_test(model, val_dataloader, num_classes, f, log_path=None, epoch=999):
        print("Evaluate on validation set...")
        if not config["no_write"]:
            f.write("Evaluate on validation set...\n")
        model.eval()

        agg_accu = []
        agg_corrects = np.zeros(num_classes, dtype=int)
        agg_lengths = np.zeros(num_classes, dtype=int)
        misclassify_mats = torch.zeros((num_classes, num_classes), dtype=int)
        # print("len(val_dataloader)",len(val_dataloader)) # 829
        for i, data in enumerate(val_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images) 
            
            outputs = testnet.aggregate_outputs(outputs, 2) # [16, 16]
            accu = accuracy(outputs, labels)
            # avg_accu, class_accu = class_avg_accuracy(outputs, labels)
            # avg_accu, class_accu, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            corrects, lengths, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            # print("corrects, lengths:",corrects, lengths)
            
            agg_accu.append(accu)
            # agg_class_accu.append(class_accu)
            agg_corrects += corrects
            agg_lengths += lengths
            misclassify_mats += misclassify_mat

        print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
        print("Validation # corrects per class and class sizes: ", agg_corrects, agg_lengths)
        agg_class_accu = [(agg_corrects[c] / agg_lengths[c]) for c in range(num_classes)] # if lengths[c] > 0 else None
        print("Validation class accuracies:", agg_class_accu)
        print("Validation average class accuracy: ", np.mean(np.array(agg_class_accu), axis=0))
        # print("Validation misclassify_mats: ",misclassify_mats)
        misclassify_mats_normalized = misclassify_mats / misclassify_mats.sum(dim=-1).unsqueeze(-1)
        # print("Validation misclassify_mats_normalized: ", misclassify_mats_normalized)
        if not config["no_write"]:
            f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
            f.write(f'Validation # corrects per class and class sizes: {agg_corrects}, {agg_lengths}\n')
            f.write(f'Validation class accuracies: {agg_class_accu}\n')
            f.write(f'Validation average class accuracy: {np.mean(np.array(agg_class_accu), axis=0)}\n')
            f.write(f'Validation misclassify_mats: {misclassify_mats}\n')
            f.write(f'Validation misclassify_mats_normalized: {misclassify_mats_normalized}\n')
        
        if log_path is not None:
            plt.imshow(misclassify_mats_normalized, cmap="Oranges", interpolation='nearest')
            plt.colorbar()
            plt.savefig(log_path+"misclassify_mats_normalized_"+str(epoch)+".png")
            plt.clf()
    
        model.train()

    epochs = config["epochs"]

    batch_size = config["batch_size"]
    division_type=1
    num_experts=config["num_experts"]
    lr = config["lr"]

    if not config["no_write"]:
        model_path = "torch_model_checkpoints/"
        model_folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"/"
        log_path = model_path+model_folder_name
        os.mkdir(log_path)
    
        print("Write to log file: "+log_path+"config.log")
        f = open(log_path+"config.log", "a")
        config_file = config["config_file"]
        f.write(f'Read training config from file: {config_file}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'division_type: {division_type}\n')
        f.write(f'num_experts: {num_experts}\n')
        f.write(f'lr: {lr}\n')
        f.write("\n")
    else:
        f = None


    dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts)
    # dataloader = ISIC2020DataLoader(batch_size=batch_size) # 0,1
    train_dataloaders = dataloader.get_train_dataloaders()
    test_dataloader = dataloader.get_test_dataloader()
    total_num_classes = dataloader.total_num_classes
    total_classes_sizes = dataloader.total_classes_sizes
    # print(dataloader.experts_train_classes_size) # [28014, 2200, 589, 110]

    testnet = Resnet50Classifier3(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=dataloader.experts_train_classes_num).to(device)

    # CELoss weighted by class frequency
    if config["loss"]["weighted"]:
        criterions = []
        temp = np.cumsum(testnet.experts_dim)
        for idx in range(testnet.num_experts):
            class_freq_weights = np.zeros(testnet.experts_dim[idx]+1)
            start = 0
            if idx >= 1:
                start = temp[idx-1]
            for i in range(len(total_classes_sizes)):
                if i >= start and i < temp[idx]:
                    class_freq_weights[i-start] = total_classes_sizes[i]
                else:
                    class_freq_weights[-1] += total_classes_sizes[i]
            # print("one:",class_freq_weights)
            class_freq_weights = np.reciprocal(class_freq_weights, dtype=np.float32)
            class_freq_weights /= np.linalg.norm(class_freq_weights)
            # print("two:",class_freq_weights)
            # one: [29091.  5953.  3417.  1562.  4149.]
# two: [0.04502016 0.22000359 0.38328397 0.8384644  0.3156619 ]
# one: [ 1100.   929.   688.   428. 41027.]
# two: [0.29406172 0.3481893  0.4701568  0.75576603 0.00788427]
# one: [  264.   259.   222.    99. 43328.]
# two: [0.30766606 0.31360555 0.36587316 0.82044286 0.00187463]
# one: [9.1000e+01 3.1000e+01 2.2000e+01 1.6000e+01 4.4012e+04]
# two: [1.3010709e-01 3.8192725e-01 5.3817022e-01 7.3998404e-01 2.6901174e-04]
            class_freq_weights = torch.tensor(class_freq_weights*10).to(device)
            # class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=np.float32) * 100).to(device)
            # criterions.append(nn.NLLLoss(weight = class_freq_weights))
            criterions.append(nn.CrossEntropyLoss(weight = class_freq_weights))
    else:
        criterions = []
        for idx in range(testnet.num_experts):
            criterions.append(nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(testnet.parameters(), lr=lr)

    if config["load_from_checkpoint"]:
        path = config["load_from_checkpoint"]
        checkpoint = torch.load(path)
        testnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print(f'Loaded model from checkpoint {path}')
        if not config["no_write"]:
            f.write(f'Loaded model from checkpoint {path}\n')
    
    if config["eval"]:
        testnet.eval()
        eval_test(testnet, test_dataloader, total_num_classes, f)
    else:
        testnet.train()
        for epoch in tqdm(range(epochs)):
            print(f'Epoch: {epoch}')
            if not config["no_write"]:
                f = open(log_path+"config.log", "a")
                f.write(f'Epoch: {epoch}\n')
            num_switches = 4
            experts_finished = [False] * num_experts
            iters = [iter(train_dataloaders[idx]) for idx in range(num_experts)]

            idx = 0
            while not all(experts_finished):
                print("\nexpert idx:",idx)
                if not config["no_write"]:
                    f.write(f'\nexpert idx: {idx}\n')
                if experts_finished[idx]:
                    idx = (idx+1)%num_experts
                    break
                
                training_losses = []
                testnet.train_one_expert(idx)
                for i in range(len(train_dataloaders[idx]) // num_switches + 1):
                    data = next(iters[idx], None)
                    if data==None:
                        experts_finished[idx] = True
                        print("experts_finished, idx=", idx)
                        if not config["no_write"]:
                            f.write(f'experts_finished, idx = {idx}\n')
                        if num_experts > 1:
                            idx = (idx+random.randint(1,num_experts-1))%num_experts
                        break
                    
                    images, labels = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = testnet(images)
                    
                    # out_probs = testnet.aggregate_outputs(outputs, 2)
                    loss = 0
                    for idx in range(testnet.num_experts):
                        # print(idx, labels, outputs[idx], testnet.convert_targets_expert(idx,labels))
                        loss += criterions[idx](outputs[idx], testnet.convert_targets_expert(idx,labels))
                    
                    training_losses.append(loss.item())

                    loss.backward()
                    optimizer.step()
                
                if len(training_losses) > 0:
                    avg_train_loss = sum(training_losses)/len(training_losses)
                    print("Average training loss:", avg_train_loss)
                    if not config["no_write"]:
                        f.write(f'Average training loss: {avg_train_loss}\n')
                
                    eval_test(testnet, test_dataloader, total_num_classes, f)
                
                # break # test network can overfit to expert3

            if not config["no_write"] and epoch%1 == 0:
                torch.save({
                    # 'epoch': epoch,
                    'model_state_dict': testnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': avg_train_loss,
                    }, f'{log_path}testnet_epoch{epoch}.pt')
                print(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt')
                f.write(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt\n')
            if not config["no_write"]:
                f.write("\n")
                f.close()
                





#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_together_dataloader(config):
    assert config["model"] == "Resnet50Classifier3"
    print("Resnet50Classifier3333333333333333")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # def eval_test(model, val_dataloader, num_classes, f, log_path=None):
    #     print("Evaluate on validation set...")
    #     if not config["no_write"]:
    #         f.write("Evaluate on validation set...\n")
    #     model.eval()

    #     agg_accu = []
    #     agg_avg_accu = []
    #     agg_class_accu = []
    #     misclassify_mats = torch.zeros((num_classes, num_classes), dtype=int)
    #     # print("len(val_dataloader)",len(val_dataloader)) # 829
    #     for i, data in enumerate(val_dataloader):
    #         images, labels = data[0].to(device), data[1].to(device)

    #         outputs = model(images) 
    #         # print("1::",outputs)
    #         outputs = testnet.aggregate_outputs(outputs, 2)
    #         # print("2::",outputs)
    #         accu = accuracy(outputs, labels)
    #         # avg_accu, class_accu = class_avg_accuracy(outputs, labels)
    #         avg_accu, class_accu, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
    #         # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            
    #         agg_accu.append(accu)
    #         agg_avg_accu.append(avg_accu)
    #         agg_class_accu.append(class_accu)
    #         misclassify_mats += misclassify_mat

    #     print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
    #     print("Validation average class accuracy: ", sum(agg_avg_accu)/len(agg_avg_accu))
    #     print("Validation class accuracies: ", np.mean(np.array(agg_class_accu), axis=0))
    #     # print("Validation misclassify_mats: ",misclassify_mats)
    #     misclassify_mats_normalized = misclassify_mats / misclassify_mats.sum(dim=-1).unsqueeze(-1)
    #     print("Validation misclassify_mats_normalized: ", misclassify_mats_normalized)
    #     if not config["no_write"]:
    #         f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
    #         f.write(f'Validation average class accuracy: {sum(agg_avg_accu)/len(agg_avg_accu)}\n')
    #         f.write(f'Validation class accuracies: {np.mean(np.array(agg_class_accu), axis=0)}\n')
    #         f.write(f'Validation misclassify_mats: {misclassify_mats}\n')
    #         f.write(f'Validation misclassify_mats_normalized: {misclassify_mats_normalized}\n')
        
    #     if log_path is not None:
    #         plt.imshow(misclassify_mats_normalized, cmap="Oranges", interpolation='nearest')
    #         plt.colorbar()
    #         plt.savefig(log_path+"misclassify_mats_normalized.png")
    
    #     model.train()
    #     return agg_accu, agg_avg_accu, agg_class_accu, misclassify_mats
    def eval_test(model, val_dataloader, num_classes, f, log_path=None, epoch=999):
        print("Evaluate on validation set...")
        if not config["no_write"]:
            f.write("Evaluate on validation set...\n")
        model.eval()

        agg_accu = []
        agg_corrects = np.zeros(num_classes, dtype=int)
        agg_lengths = np.zeros(num_classes, dtype=int)
        misclassify_mats = torch.zeros((num_classes, num_classes), dtype=int)
        # print("len(val_dataloader)",len(val_dataloader)) # 829
        for i, data in enumerate(val_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images) 
            
            outputs = testnet.aggregate_outputs(outputs, 2)
            accu = accuracy(outputs, labels)
            # avg_accu, class_accu = class_avg_accuracy(outputs, labels)
            # avg_accu, class_accu, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            corrects, lengths, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            # print("corrects, lengths:",corrects, lengths)
            
            agg_accu.append(accu)
            # agg_class_accu.append(class_accu)
            agg_corrects += corrects
            agg_lengths += lengths
            misclassify_mats += misclassify_mat

        print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
        print("Validation # corrects per class and class sizes: ", agg_corrects, agg_lengths)
        agg_class_accu = [(agg_corrects[c] / agg_lengths[c]) for c in range(num_classes)] # if lengths[c] > 0 else None
        print("Validation class accuracies:", agg_class_accu)
        print("Validation average class accuracy: ", np.mean(np.array(agg_class_accu), axis=0))
        # print("Validation misclassify_mats: ",misclassify_mats)
        misclassify_mats_normalized = misclassify_mats / misclassify_mats.sum(dim=-1).unsqueeze(-1)
        # print("Validation misclassify_mats_normalized: ", misclassify_mats_normalized)
        if not config["no_write"]:
            f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
            f.write(f'Validation # corrects per class and class sizes: {agg_corrects}, {agg_lengths}\n')
            f.write(f'Validation class accuracies: {agg_class_accu}\n')
            f.write(f'Validation average class accuracy: {np.mean(np.array(agg_class_accu), axis=0)}\n')
            f.write(f'Validation misclassify_mats: {misclassify_mats}\n')
            f.write(f'Validation misclassify_mats_normalized: {misclassify_mats_normalized}\n')
        
        if log_path is not None:
            plt.imshow(misclassify_mats_normalized, cmap="Oranges", interpolation='nearest')
            plt.colorbar()
            plt.savefig(log_path+"misclassify_mats_normalized_"+str(epoch)+".png")
            plt.clf()
    
        model.train()
    

    epochs = config["epochs"]

    batch_size = config["batch_size"]
    division_type=1
    num_experts=config["num_experts"]
    lr = config["lr"]

    if not config["no_write"]:
        model_path = "torch_model_checkpoints/"
        model_folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"/"
        log_path = model_path+model_folder_name
        os.mkdir(log_path)
    
        print("Write to log file: "+log_path+"config.log")
        f = open(log_path+"config.log", "a")
        config_file = config["config_file"]
        f.write(f'Read training config from file: {config_file}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'division_type: {division_type}\n')
        f.write(f'num_experts: {num_experts}\n')
        f.write(f'lr: {lr}\n')
        f.write("\n")
    else:
        f = None


    ## dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts)
    # dataloader = ISIC2020DataLoader(batch_size=batch_size) # 0,1
    if "datasetmultiple" in config:
        dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=0, num_experts=1, dataset = config["datasetmultiple"]) # dataset = "orig", "multiple3"
    else:
        dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=0, num_experts=1) # 0,1: not separate dataloader
    train_dataloader = dataloader.get_train_dataloaders()[0]
    test_dataloader = dataloader.get_test_dataloader()
    total_num_classes = dataloader.total_num_classes
    total_classes_sizes = dataloader.total_classes_sizes
    # print(dataloader.experts_train_classes_size) # [28014, 2200, 589, 110]

    experts_train_classes_num = get_expert_num_classes(total_num_classes, division_type = 1, num_experts = num_experts)
    # testnet = Resnet50Classifier3(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=experts_train_classes_num).to(device)

    if "pretrained_extractor" in config:
        testnet = Resnet50Classifier3(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=experts_train_classes_num, pretrained_extractor=config["pretrained_extractor"]).to(device)
    else:
        print("no pretrained extractor")
        testnet = Resnet50Classifier3(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=experts_train_classes_num).to(device)
    
    if config["loss"]["type"] == "focal_loss" and not config["loss"]["weighted"]:
        print("unweighted focal lossssssss")
        # criterion = focal_loss(alpha = torch.tensor([1/total_num_classes]*total_num_classes).to(device),
        #        gamma = 2,
        #        reduction = 'mean',
        #        device=device)
        criterions = []
        for idx in range(testnet.num_experts):
            criterions.append(focal_loss(
                alpha = torch.tensor([15/16,1/16]).to(device),
                gamma = 2,
                reduction = 'mean',
                device=device))
    elif config["loss"]["type"] == "focal_loss" and config["loss"]["weighted"]: # not implemented / used
        # class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=np.float32))
        # class_freq_weights /= class_freq_weights.sum()
        # class_freq_weights = class_freq_weights.to(device)
        # criterion = focal_loss(alpha = class_freq_weights,
        #        gamma = 2,
        #        reduction = 'mean',
        #        device=device)
        criterions = []
        for idx in range(testnet.num_experts):
            criterions.append(nn.CrossEntropyLoss())
    elif config["loss"]["type"] == "CELoss" and config["loss"]["weighted"]:
        print("weighted CELossssssss")
        criterions = []
        temp = np.cumsum(testnet.experts_dim)
        for idx in range(testnet.num_experts):
            class_freq_weights = np.zeros(testnet.experts_dim[idx]+1)
            start = 0
            if idx >= 1:
                start = temp[idx-1]
            for i in range(len(total_classes_sizes)):
                if i >= start and i < temp[idx]:
                    class_freq_weights[i-start] = total_classes_sizes[i]
                else:
                    class_freq_weights[-1] += total_classes_sizes[i]
            class_freq_weights = np.reciprocal(class_freq_weights, dtype=np.float32)
            class_freq_weights /= np.linalg.norm(class_freq_weights)
            class_freq_weights = torch.tensor(class_freq_weights).to(device)
            # class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=np.float32) * 100).to(device)
            # criterions.append(nn.NLLLoss(weight = class_freq_weights))
            criterions.append(nn.CrossEntropyLoss(weight = class_freq_weights))
    elif config["loss"]["type"] == "CELoss" and not config["loss"]["weighted"]:
        print("unweighted CELossssssss")
        criterions = []
        for idx in range(testnet.num_experts):
            criterions.append(nn.CrossEntropyLoss())
    
    # # CELoss weighted by class frequency
    # if config["loss"]["weighted"]:
    #     print("weighted CELossssssss")
    #     criterions = []
    #     temp = np.cumsum(testnet.experts_dim)
    #     for idx in range(testnet.num_experts):
    #         class_freq_weights = np.zeros(testnet.experts_dim[idx]+1)
    #         start = 0
    #         if idx >= 1:
    #             start = temp[idx-1]
    #         for i in range(len(total_classes_sizes)):
    #             if i >= start and i < temp[idx]:
    #                 class_freq_weights[i-start] = total_classes_sizes[i]
    #             else:
    #                 class_freq_weights[-1] += total_classes_sizes[i]
    #         class_freq_weights = np.reciprocal(class_freq_weights, dtype=np.float32)
    #         class_freq_weights /= np.linalg.norm(class_freq_weights)
    #         class_freq_weights = torch.tensor(class_freq_weights).to(device)
    #         criterions.append(nn.CrossEntropyLoss(weight = class_freq_weights))
    # else:
    #     print("unweighted CELossssssss")
    #     criterions = []
    #     for idx in range(testnet.num_experts):
    #         criterions.append(nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(testnet.parameters(), lr=lr)

    if config["load_from_checkpoint"]:
        path = config["load_from_checkpoint"]
        checkpoint = torch.load(path)
        testnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print(f'Loaded model from checkpoint {path}')
        if not config["no_write"]:
            f.write(f'Loaded model from checkpoint {path}\n')
    
    if config["eval"]:
        testnet.eval()
        eval_test(testnet, test_dataloader, total_num_classes, f, log_path)
    else:
        testnet.train()
        eval_every = len(train_dataloader) // 4 # 318
        for epoch in tqdm(range(epochs)):
            print(f'Epoch: {epoch}')
            if not config["no_write"]:
                f = open(log_path+"config.log", "a")
                f.write(f'Epoch: {epoch}\n')
                
            training_losses = []
            for i, data in enumerate(train_dataloader):   
                images, labels = data[0].to(device), data[1].to(device)
                # print("labels",torch.unique(labels))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = testnet(images)
                
                # preds = testnet.aggregate_outputs(outputs, 2)
                loss = 0
                for idx in range(testnet.num_experts):
                    # print(idx, labels, outputs[idx], testnet.convert_targets_expert(idx,labels))
                    # print(idx, outputs[idx])
                    # print(labels, testnet.convert_targets_expert(idx,labels))
                    loss += criterions[idx](outputs[idx], testnet.convert_targets_expert(idx,labels))
                # print("this loss:",loss)
                training_losses.append(loss.item())

                loss.backward()
                optimizer.step()
            
                if i % eval_every == eval_every-1:
                    if len(training_losses) > 0:
                        avg_train_loss = sum(training_losses)/len(training_losses)
                        print("Average training loss:", avg_train_loss)
                        if not config["no_write"]:
                            f.write(f'Average training loss: {avg_train_loss}\n')
                    
                        eval_test(testnet, test_dataloader, total_num_classes, f)

            if not config["no_write"] and epoch%1 == 0:
                torch.save({
                    'model_state_dict': testnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'{log_path}testnet_epoch{epoch}.pt')
                print(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt')
                f.write(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt\n')
            if not config["no_write"]:
                f.write("\n")
                f.close()







#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_together_dataloader_pretrained_extractor(config):
    assert config["model"] == "Resnet50Classifier4" and config["loss"]["type"] == "CELoss"
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # def eval_test(model, val_dataloader, num_classes, f, log_path=None, epoch=-1):
    #     print("Evaluate on validation set...")
    #     if not config["no_write"]:
    #         f.write("Evaluate on validation set...\n")
    #     model.eval()

    #     agg_accu = []
    #     agg_avg_accu = []
    #     agg_class_accu = []
    #     misclassify_mats = torch.zeros((num_classes, num_classes), dtype=int)
    #     # print("len(val_dataloader)",len(val_dataloader)) # 829
    #     for i, data in enumerate(val_dataloader):
    #         images, labels = data[0].to(device), data[1].to(device)

    #         outputs = model(images) 
            
    #         outputs = testnet.aggregate_outputs(outputs, 2)
    #         accu = accuracy(outputs, labels)
    #         # avg_accu, class_accu = class_avg_accuracy(outputs, labels)
    #         avg_accu, class_accu, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
    #         # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            
    #         agg_accu.append(accu)
    #         agg_avg_accu.append(avg_accu)
    #         agg_class_accu.append(class_accu)
    #         misclassify_mats += misclassify_mat

    #     print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
    #     print("Validation average class accuracy: ", sum(agg_avg_accu)/len(agg_avg_accu))
    #     print("Validation class accuracies: ", np.mean(np.array(agg_class_accu), axis=0))
    #     # print("Validation misclassify_mats: ",misclassify_mats)
    #     misclassify_mats_normalized = misclassify_mats / misclassify_mats.sum(dim=-1).unsqueeze(-1)
    #     # print("Validation misclassify_mats_normalized: ", misclassify_mats_normalized)
    #     if not config["no_write"]:
    #         f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
    #         f.write(f'Validation average class accuracy: {sum(agg_avg_accu)/len(agg_avg_accu)}\n')
    #         f.write(f'Validation class accuracies: {np.mean(np.array(agg_class_accu), axis=0)}\n')
    #         f.write(f'Validation misclassify_mats: {misclassify_mats}\n')
    #         f.write(f'Validation misclassify_mats_normalized: {misclassify_mats_normalized}\n')
        
    #     if log_path is not None:
    #         plt.imshow(misclassify_mats_normalized, cmap="Oranges", interpolation='nearest')
    #         plt.colorbar()
    #         plt.savefig(log_path+"misclassify_mats_normalized_"+str(epoch)+".png")
    #         plt.clf()
    
    #     model.train()
    #     return agg_accu, agg_avg_accu, agg_class_accu, misclassify_mats

    def eval_test(model, val_dataloader, num_classes, f, log_path=None, epoch=999):
        print("Evaluate on validation set...")
        if not config["no_write"]:
            f.write("Evaluate on validation set...\n")
        model.eval()

        agg_accu = []
        agg_corrects = np.zeros(num_classes, dtype=int)
        agg_lengths = np.zeros(num_classes, dtype=int)
        misclassify_mats = torch.zeros((num_classes, num_classes), dtype=int)
        # print("len(val_dataloader)",len(val_dataloader)) # 829
        for i, data in enumerate(val_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images) 
            
            outputs = testnet.aggregate_outputs(outputs, 2) # [16, 16]
            accu = accuracy(outputs, labels)
            # avg_accu, class_accu = class_avg_accuracy(outputs, labels)
            # avg_accu, class_accu, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            corrects, lengths, misclassify_mat = class_avg_accuracy_misclassifymat(outputs, labels)
            # print("accuracy = ",accu, ", avg_class_accuracy = ", avg_accu, ", class_accuracy: ",class_accu)
            # print("corrects, lengths:",corrects, lengths)
            
            agg_accu.append(accu)
            # agg_class_accu.append(class_accu)
            agg_corrects += corrects
            agg_lengths += lengths
            misclassify_mats += misclassify_mat

        print("Validation accuracy: ", sum(agg_accu)/len(agg_accu))
        print("Validation # corrects per class and class sizes: ", agg_corrects, agg_lengths)
        agg_class_accu = [(agg_corrects[c] / agg_lengths[c]) for c in range(num_classes)] # if lengths[c] > 0 else None
        print("Validation class accuracies:", agg_class_accu)
        print("Validation average class accuracy: ", np.mean(np.array(agg_class_accu), axis=0))
        # print("Validation misclassify_mats: ",misclassify_mats)
        misclassify_mats_normalized = misclassify_mats / misclassify_mats.sum(dim=-1).unsqueeze(-1)
        # print("Validation misclassify_mats_normalized: ", misclassify_mats_normalized)
        if not config["no_write"]:
            f.write(f'Validation accuracy: {sum(agg_accu)/len(agg_accu)}\n')
            f.write(f'Validation # corrects per class and class sizes: {agg_corrects}, {agg_lengths}\n')
            f.write(f'Validation class accuracies: {agg_class_accu}\n')
            f.write(f'Validation average class accuracy: {np.mean(np.array(agg_class_accu), axis=0)}\n')
            f.write(f'Validation misclassify_mats: {misclassify_mats}\n')
            f.write(f'Validation misclassify_mats_normalized: {misclassify_mats_normalized}\n')
        
        if log_path is not None:
            plt.imshow(misclassify_mats_normalized, cmap="Oranges", interpolation='nearest')
            plt.colorbar()
            plt.savefig(log_path+"misclassify_mats_normalized_"+str(epoch)+".png")
            plt.clf()
    
        model.train()

    epochs = config["epochs"]

    batch_size = config["batch_size"]
    division_type=1
    num_experts=config["num_experts"]
    lr = config["lr"]

    if not config["no_write"]:
        model_path = "torch_model_checkpoints/"
        model_folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"/"
        log_path = model_path+model_folder_name
        os.mkdir(log_path)
    
        print("Write to log file: "+log_path+"config.log")
        f = open(log_path+"config.log", "a")
        config_file = config["config_file"]
        f.write(f'Read training config from file: {config_file}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'division_type: {division_type}\n')
        f.write(f'num_experts: {num_experts}\n')
        f.write(f'lr: {lr}\n')
        f.write("\n")
    else:
        f = None


    # dataloader = ISIC2020DataLoader(batch_size=batch_size, division_type=division_type, num_experts=num_experts)
    dataloader = ISIC2020DataLoader(batch_size=batch_size) # 0,1
    train_dataloader = dataloader.get_train_dataloaders()[0]
    test_dataloader = dataloader.get_test_dataloader()
    total_num_classes = dataloader.total_num_classes
    total_classes_sizes = dataloader.total_classes_sizes
    # print(dataloader.experts_train_classes_size) # [28014, 2200, 589, 110]

    experts_train_classes_num = get_expert_num_classes(total_num_classes, division_type = 1, num_experts = num_experts)
    testnet = Resnet50Classifier4(output_dimension=total_num_classes, num_experts=num_experts, experts_dim=experts_train_classes_num, pretrained_extractor=config["pretrained_extractor"]).to(device)

    if config["loss"]["type"] == "CELoss" and config["loss"]["weighted"]:
        criterions = []
        temp = np.cumsum(testnet.experts_dim)
        for idx in range(testnet.num_experts):
            class_freq_weights = np.zeros(testnet.experts_dim[idx]+1)
            start = 0
            if idx >= 1:
                start = temp[idx-1]
            for i in range(len(total_classes_sizes)):
                if i >= start and i < temp[idx]:
                    class_freq_weights[i-start] = total_classes_sizes[i]
                else:
                    class_freq_weights[-1] += total_classes_sizes[i]
            class_freq_weights = np.reciprocal(class_freq_weights, dtype=np.float32)
            class_freq_weights /= np.linalg.norm(class_freq_weights)
            class_freq_weights = torch.tensor(class_freq_weights).to(device)
            # class_freq_weights = torch.tensor(np.reciprocal(total_classes_sizes, dtype=np.float32) * 100).to(device)
            # criterions.append(nn.NLLLoss(weight = class_freq_weights))
            criterions.append(nn.CrossEntropyLoss(weight = class_freq_weights))
    elif config["loss"]["type"] == "CELoss" and not config["loss"]["weighted"]:
        criterions = []
        for idx in range(testnet.num_experts):
            criterions.append(nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(testnet.parameters(), lr=lr)

    if config["load_from_checkpoint"]:
        path = config["load_from_checkpoint"]
        checkpoint = torch.load(path)
        testnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print(f'Loaded model from checkpoint {path}')
        if not config["no_write"]:
            f.write(f'Loaded model from checkpoint {path}\n')
    
    if config["eval"]:
        testnet.eval()
        eval_test(testnet, test_dataloader, total_num_classes, f, log_path)
    else:
        testnet.train()
        eval_every = len(train_dataloader) // 8
        for epoch in tqdm(range(epochs)):
            print(f'Epoch: {epoch}')
            if not config["no_write"]:
                f = open(log_path+"config.log", "a")
                f.write(f'Epoch: {epoch}\n')
                
            training_losses = []
            for i, data in enumerate(train_dataloader):   
                images, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = testnet(images)
                
                # out_probs = testnet.aggregate_outputs(outputs, 2)
                loss = 0
                for idx in range(testnet.num_experts):
                    # print(idx, labels, outputs[idx], testnet.convert_targets_expert(idx,labels))
                    loss += criterions[idx](outputs[idx], testnet.convert_targets_expert(idx,labels))
                # print("this loss:",loss)
                
                training_losses.append(loss.item())

                loss.backward()
                optimizer.step()
            
                if i % eval_every == eval_every-1:
                    if len(training_losses) > 0:
                        avg_train_loss = sum(training_losses)/len(training_losses)
                        print("Average training loss:", avg_train_loss)
                        if not config["no_write"]:
                            f.write(f'Average training loss: {avg_train_loss}\n')
                    
                        if not config["no_write"]:
                            eval_test(testnet, test_dataloader, total_num_classes, f, log_path, epoch=epoch)
                        else:
                            eval_test(testnet, test_dataloader, total_num_classes, f, None, epoch=epoch)
                        # _,_,_,_ = eval_test(testnet, test_dataloader, total_num_classes, f, log_path, epoch=epoch)

            if not config["no_write"] and epoch%1 == 0:
                torch.save({
                    'model_state_dict': testnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'{log_path}testnet_epoch{epoch}.pt')
                print(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt')
                f.write(f'Model general checkpoint saved as: {log_path}testnet_epoch{epoch}.pt\n')
            if not config["no_write"]:
                f.write("\n")
                f.close()