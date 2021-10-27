# CMKD-MINDS: Learning an Augmented sEMG Representation via Cross-modality Knowledge Distillation
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

This is an official pytorch implementation of [Learning an Augmented sEMG Representation via Cross-modality Knowledge Distillation (under review)]()

* Reproduced neural networks for comparison:
  * [x] [Multi Stream CNN(WeiNet)](https://www.sciencedirect.com/science/article/abs/pii/S0167865517304439)
  * [x] [EUNet](https://github.com/increase24/EUNet)
  * [x] [XceptionTime](https://arxiv.org/abs/1911.03803)
  * [x] [Multi-scale Kernel CNN(MKCNN)](https://ieeexplore.ieee.org/document/9495836)
  * [ ] ResEUNet

## Environment
The code is developed using python 3.7 on Ubuntu 20.04. NVIDIA GPU is needed.


## Data preparing
The complete hybrid sEMG/AUS dataset is not released now. We apply collected sEMG/AUS data of one subject for code testing, which can be downloaded from: [Baidu Disk](https://pan.baidu.com/s/1qitEFqvwPmD20HnbqgsDcg)
(code: h99k).

Your directory tree should look like this: 
```
${ROOT}/data
├── EMG
|   |—— s1_***_EMG.txt
|   |—— s2_***_EMG.txt
|   |   ...
|   └── s8_***_EMG.txt
└── US
    |—— s1_***_US.txt
    |—— s2_***_US.txt
    |   ...
    └── s8_***_US.txt
```

## Usage
### Installation
1. Clone this repo
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
### Training
For training a network on single sEMG or AUS modality, run the script files `train_emg/us_***.sh` in the directory `./scripts/`. For instance:
```
# train network XceptionTime on sEMG modality
sh scripts/train_emg_XceptionTime.sh

# train network EUNet on US modality
sh scripts/train_us_EUNet.sh
```
### Evaluation
  ```
  
  ```

## Results

## Citation
If you find this repository useful for your research, please cite with:
```
```

## Contact
If you have any questions, feel free to contact me through jia.zeng@sjtu.edu.cn or Github issues.

