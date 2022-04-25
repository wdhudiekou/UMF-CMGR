# UMF-CMGR
 

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/CDVD-TSP/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)



# Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration [IJCAI2022 Oral Presentation]
#### [Paper]()
## Abstract
Recent learning-based image fusion methods have marked numerous progress in pre-registered multi-modality data, but suffered serious ghosts dealing with misaligned multi-modality data, due to the spatial deformation and the difficulty narrowing cross-modality discrepancy.
To overcome the obstacles, in this paper, we present a robust cross-modality generation-registration paradigm for unsupervised misaligned infrared and visible image fusion (IVIF).
Specifically, we propose a Cross-modality Perceptual Style Transfer Network (CPSTN) to generate a pseudo infrared image taking a visible image as input.
Benefiting from the favorable geometry preservation ability of the CPSTN, the generated pseudo infrared image embraces a sharp structure, which is more conducive to transforming cross-modality image alignment into mono-modality registration coupled with the structure-sensitive of the infrared image.
In this case, we introduce a Multi-level Refinement Registration Network (MRRN) to predict the displacement vector field between distorted and pseudo infrared images and reconstruct registered infrared image under the mono-modality setting.
Moreover, to better fuse the registered infrared images and visible images, we present a feature Interaction Fusion Module (IFM) to adaptively select more meaningful features for fusion in the Dual-path Interaction Fusion Network (DIFN).
Extensive experimental results suggest that the proposed method performs superior capability on misaligned cross-modality image fusion.

<div align=center>
<img src="https://github.com/wdhudiekou/UMF-CMGR/tree/main/Fig/network.png" width="80%">

Fig. 1：The workflow of the proposed unsupervised cross-modality fusion network for misaligned infrared and visible images.
</div>

## Experimental Results
Coming soon...


## Citation
```
@InProceedings{Wang_2022_IJCAI,
	author = {Di, Wang and Jinyuan, Liu and Xin, Fan and Risheng Liu},
	title = {Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration},
	booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
	year = {2022}
}
```