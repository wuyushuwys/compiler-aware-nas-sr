# [ECCV2022] Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution

Pytorch Implementation of  [Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution](https://arxiv.org/abs/2207.12577)


## Usage

### Dependencies

```bash
conda install -y pytorch==1.9.1 torchvision cudatoolkit=11.1 -c pytorch -c conda-forge -c nvidia
conda install -y tensorboard h5py scikit-image
```

### Train

- Configuration
    - dataset (default: div2k): train dataset
    - eval_datasets (set5/set14/urban100/bsds100...): evaluation dataset
    - scale (4): scale factor
    - num_blocks (default: 16): number of blocks in wdsr
    - num_residual_units (default 24): number of residual units in wdsr
1. Download the dataset, put them in folder `data`
2. Prepare dataset using `prepare_dataset.py` before distributed training
3. Training
   1. Pretrain
      1. Using `pretraining.bash` to train a pretrained model. Two pretrained weights have already offered [here](models/pretrained_weights)
   2. Search
       - You may modify the weights [here](loss_config.py) to fine-tune search results
       - Multi GPUs Training
           - modify configuration in `train.bash` then run `bash train.bash <log path (optional)>`
- Speed model
  - Here are several trained speed models provided [here](speed_models/weights) for different feature size and platform


## Datasets

[DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[Benchmarks (Set5, BSDS100, Urban100)](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip)

Download and organize data like:

```bash
./data/DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X2
    └── X3
    └── X4
./data/Set5/*.png
./data/BSDS100/*.png
./data/Urban100/*.png
```

## Acknowledgements
[https://github.com/ychfan/wdsr](https://github.com/ychfan/wdsr)



## Citation

If you find this code useful for your research, please cite our paper

```
@misc{https://doi.org/10.48550/arxiv.2207.12577,

  author = {Wu, Yushu and Gong, Yifan and Zhao, Pu and Li, Yanyu and Zhan, Zheng and Niu, Wei and Tang, Hao and Qin, Minghai and Ren, Bin and Wang, Yanzhi},
  title = {Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution},
  doi = {10.48550/ARXIV.2207.12577},
  url = {https://arxiv.org/abs/2207.12577},
  publisher = {arXiv},
  year = {2022},
}
```