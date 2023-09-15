

conda activate py3.9-torch
cd ~/exp1/scripts3/my

python train.py --config="configs/config_isic_baseline.json" --resume="torch_model_checkpoints/2023_05_01_20_13_41/testnet_epoch37.pt" --eval


python train.py --config="configs/config_isic5.3.json" --resume="torch_model_checkpoints/2023_05_04_20_46_41/testnet_epoch49.pt" --eval
python train.py --config="configs/config_isic5.4.json" --resume="torch_model_checkpoints/2023_05_06_17_26_06/testnet_epoch45.pt" --eval

python train.py --config="configs/config_isic7.1.json" --resume="torch_model_checkpoints/2023_05_06_17_36_25/testnet_epoch45.pt" --eval
python train.py --config="configs/config_isic7.2.json" --resume="torch_model_checkpoints/2023_05_05_22_42_28/testnet_epoch32.pt" --eval
python train.py --config="configs/config_isic7.2.json" --resume="torch_model_checkpoints/2023_05_06_17_37_05/testnet_epoch45.pt" --eval

python train.py --config="configs/config_isic7.3.json" --resume="torch_model_checkpoints/2023_05_07_16_12_25/testnet_epoch4.pt" --eval
python train.py --config="configs/config_isic7.3.json" --resume="torch_model_checkpoints/2023_05_07_16_12_25/testnet_epoch49.pt" --eval

python train.py --config="configs/config_isic4.json" --resume="torch_model_checkpoints/2023_05_03_14_47_33/testnet_epoch14.pt" --eval
python train.py --config="configs/config_isic4.json" --resume="torch_model_checkpoints/2023_05_03_14_47_33/testnet_epoch50.pt" --eval

python train.py --config="configs/config_isic4.1.json" --resume="torch_model_checkpoints/2023_05_04_20_46_02/testnet_epoch44.pt" --eval
python train.py --config="configs/config_isic4.2.json" --resume="torch_model_checkpoints/2023_05_04_20_48_26/testnet_epoch20.pt" --eval
python train.py --config="configs/config_isic4.2.json" --resume="torch_model_checkpoints/2023_05_04_20_48_26/testnet_epoch47.pt" --eval

python train.py --config="configs/config_isic5.json" --resume="torch_model_checkpoints/2023_05_03_12_18_28/testnet_epoch49.pt" --eval

python train.py --config="configs/config_isic6.1.json" --resume="torch_model_checkpoints/2023_05_05_16_25_42/testnet_epoch47.pt" --eval
