# Vision-Based Dynamics Prediction Under Environment Misalignment Challenge (VDP-EMC)

This repo contains the dataset proposed by the paper: 

"**A Critical View of Vision-Based Long-Term Dynamics Prediction Under Environment Misalignment**", Hanchen Xie, Jiageng Zhu, Mahyar Khayatkhoei, Jiazhi Li, Mohamed E Hussein, Wael AbdAlmageed, ICML 2023. [Paper](https://arxiv.org/abs/2305.07648)

Most of the codes are from RPCIN repository (https://github.com/HaozhiQi/RPIN), and we provide them here as an example of using the datasets. We sincerely appreciate RPCIN authors for their outstanding work and code base. Please refer to the original RPCIN repo for details of RPCIN and running it on other dynamics prediction datasets. 

## Datasets
In the paper, to investigate the environment misalignment challenges, we proposed four datasets: *SimB-Border*, *SimB-Split*, *BlenB-Border*, and *BlenB-Split*. Please use the following command to download the dataset:

*SimB-Border*:

`gdown 1ws0RoFRJKC2hdfDFTWcUqfqWjrBpb3L1`

*SimB-Split*:

`gdown 1pUERqLu_LtICYbjIj1Xxik7pxVOVHAd1`

*BlenB-Border*:

`gdown 1YE9qYrYhLi7XZPpat2Ad5w7yOdHf4Lgp`

*BlenB-Split*:

`gdown 10r-naMspzhKI069vd3_ILyqsyc3sT3gL`

Please put the dataset under *./data* folder and untar them. Note that each of the datasets, after untar, can take a lot of space (datasets on Sim Domain can take 150G+), please ensure the available disk space is sufficient. 

## Dataset Hierarchy:
```
Dataset_Name (e.g., SimB-Split)
|-train
    |---00000 (video_name)
      |---00000_bmask.pkl (environment mask)
      |---00000_data.pkl (data file, Sim Domain Only)
      |---00000_debug.png (data visualization, Sim Domain Only)
      |---00000.png (data file, Blen Domain Only)
             .
             .
             .
    |---00001
    |---00002
        .
        .
        .
    |---00000.pkl (labels for video “00000”)
    |---00001.pkl
        .
        .
        .
|-test (same with "train")
|-train_env_meta.pkl (environment meta for rendering blenb, Sim Domain Only)
|-test_env_meta.pkl (environment meta for rendering blenb, Sim Domain Only)
```

## Package Requirment
We ran experiments with python 3.9, PyTorch 1.10.1, and cuda 11.3. Haven't tested on PyTorch 2.0+ yet. 

## Train Network
Please use following command as template for training (It will use all CUDA_VISIBLE_DEVICES):

`python train.py --cfg ./configs/simb_split/rpcin_bn.yaml --output simb_split_bn`

## Evaluation

Please use following commands as templates for evaluation:

Environment Aligned:

`python test.py --cfg ./configs/simb_split/rpcin_bn.yaml --predictor-init ./outputs/phys/simb_split/simb_split_bn/ckpt_best.path.tar`

Cross-Domain Challenge (Note the dataset domain difference: *SimB-Split*->*BlenB-Split*):

`python test.py --cfg ./configs/blenb_split/rpcin_bn.yaml --predictor-init ./outputs/phys/simb_split/simb_split_bn/ckpt_best.path.tar`

Cross-Context Challenge (Note the dataset context difference: *SimB-Split*->*SimB-Border*):

`python test.py --cfg ./configs/simb_border/rpcin_bn.yaml --predictor-init ./outputs/phys/simb_split/simb_split_bn/ckpt_best.path.tar`

## Creating Dataset

Please use *./tools/gen_billiard_with_boundry.py* to create *SimB-Border* and *./tools/gen_billiard_split_boundry.py* to create *SimB-Split*. After creating dataset, please use *./tools/prepare_billiard.py* for extracting the .hkl file. Environment metadata for train and test are within the respected folder. Please copy and rename them (e.g., *train_env_meta.pkl*) to each dataset's root directory. They are needed for rendering datasets in *Blen* domain. 

Rendering dataset in *Blen* domain, requires ground-truth files under train/test folders (*./SimB-Border/train/00000.pkl*) and *train/test_env_meta.pkl* file under the dataset root folder (e.g., *./SimB-Border*). Blender engineer [file](https://drive.google.com/file/d/1IKjCPLdb6cClqq_A3ZRgOqmWUnHX8tqE/view?usp=sharing) need to be placed outside of the dataset root folder. Please check the engineer file for details. 

## Citing Our Work
If you found our work are helping, please consider to cite our work:

```

@InProceedings{pmlr-v202-xie23e,
  title = 	 {A Critical View of Vision-Based Long-Term Dynamics Prediction Under Environment Misalignment},
  author =       {Xie, Hanchen and Zhu, Jiageng and Khayatkhoei, Mahyar and Li, Jiazhi and Hussein, Mohamed E. and Abdalmageed, Wael},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {38258--38271},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/xie23e/xie23e.pdf},
  url = 	 {https://proceedings.mlr.press/v202/xie23e.html},
}

```
