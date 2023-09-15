import argparse
import json
# from train_helpers import train0, train_separate_expert_dataloaders, train_separate_expert_dataloaders_classifier3, train_together_dataloader, train_together_dataloader_pretrained_extractor
from train_helpers import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--eval', action='store_true', help='if --eval, just evaluate on testing set (need to provide checkpoint from --resume)')
    parser.add_argument('--nowrite', action='store_true', help='if --nowrite, do not write to config file')
    
    args = parser.parse_args()
    if not args.config:
        train0()
    else:
        print("Load from config file: ", args.config)
        with open(args.config) as f:
            config_dict = json.load(f)
        config_dict["config_file"] = args.config
        config_dict["load_from_checkpoint"] = args.resume
        config_dict["eval"] = args.eval
        config_dict["no_write"] = args.nowrite
        print(config_dict)
        if not "model" in config_dict or config_dict["model"] == "Resnet50Classifier" or config_dict["model"] == "Resnet50Classifier2":
            train_separate_expert_dataloaders(config_dict)
        else:
            if "separate_dataloaders" in config_dict and config_dict["separate_dataloaders"]:
                train_separate_expert_dataloaders_classifier3(config_dict)
            else:
                if config_dict["model"] == "Resnet50Classifier4": # "pretrained_extractor" in config_dict:
                    train_together_dataloader_pretrained_extractor(config_dict)
                else:
                    train_together_dataloader(config_dict)

