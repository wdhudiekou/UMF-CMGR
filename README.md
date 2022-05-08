# UMF
 

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/UMF-CMGR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)



# Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration [IJCAI2022 Oral Presentation]
#### [Paper]()

<div align=center>
<img src="https://github.com/wdhudiekou/UMF-CMGR/blob/main/Fig/network.png" width="80%">


</div>

## Requirements
- CUDA 10.1
- Python 3.6 (or later)
- Pytorch 1.6.0
- Torchvision 0.7.0
- OpenCV 3.4

## Data preparation
1. You can obtain deformation infrared images for training/testing process by
    ```python
       cd ./data
       python get_test_data.py
   
In 'Trainer/train_reg.py', deformation infrared images are generated in real time by default during training.

2. You can obtain self-visual saliency maps for training IVIF fusion by
    ```python
       cd ./data
       python get_svs_map.py

## Get start
1. You can use the pseudo infrared images [[link](https://pan.baidu.com/s/1M79RuHVe6udKhcJIA7yXgA) code: qqyj] generated by our CPSTN to train/test the registration process:
    ```python
       cd ./Trainer
       python train_reg.py

       cd ./Test
       python test_reg.py

2. If you tend to train Registration and Fusion processes separately, You can run following commands:      

    ```python
       cd ./Trainer
       python train_reg.py

       cd ./Trainer
       python train_fuse.py
  The corresponding test code 'test_reg.py' and 'test_fuse.py' can be found in 'Test' folder.

3. If you tend to train Registration and Fusion processes jointly, You can run following command: 
   ```python
       cd ./Trainer
       python train_reg_fusion.py

  The corresponding test code 'test_reg_fusion.py' can be found in 'Test' folder.

## Dataset
Please download the following datasets:
* RoadScene  [[link](https://github.com/hanna-xu/RoadScene)]
* TNO        [[link](http://figshare.com/articles/TNO\_Image\_Fusion\_Dataset/1008029)]

## Experimental Results

Please download the pseudo infrared images generated by our CPSTN:
* Fake_infrared_images  [[link](https://pan.baidu.com/s/1M79RuHVe6udKhcJIA7yXgA)] code: qqyj

Please download the registered infrared images by our UMF:
* Registered_results on RoadScene  [[link](https://pan.baidu.com/s/161lbmGx8TDphx0Uf9cAtfg )] code: 4cx2
* Registered_results on TNO        [[link](https://pan.baidu.com/s/1AO2T4LMsujIQcrJT9WnHpg )] code: 2edi

Please download the fused images by our UMF:

* Fused_results on RoadScene  [[link](https://pan.baidu.com/s/1aG_CI9fFIhsV2Z2ThMUPQg )] code: 1zuu
* Fused_results on TNO        [[link](https://pan.baidu.com/s/10Me7GpM_tvHgkWVzv2pv3g )] code: 22gc


## Citation
```
@InProceedings{Wang_2022_IJCAI,
	author = {Di, Wang and Jinyuan, Liu and Xin, Fan and Risheng Liu},
	title = {Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration},
	booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
	year = {2022}
}
```