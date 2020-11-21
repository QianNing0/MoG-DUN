# Accurate and Lightweight Image Super-Resolution with Model-Guided Deep Unfolding Network  [[IEEE]](https://ieeexplore.ieee.org/document/9257009) [[homepage]](https://see.xidian.edu.cn/faculty/wsdong/)


This repository is Pytorch code for our proposed MoG-DUN.

The code is developed by [RCAN](https://github.com/yulunzhang/RCAN)  and tested on Ubuntu 16.04 environment (Python 3.5/3.6/3.7, PyTorch 1.0.0/1.0.1, 9.0/10.0) with 2080Ti/1080Ti GPUs.


If you find our work useful in your research or publications, please consider citing:

```latex
@ARTICLE{9257009,
  author={Q. {Ning} and W. {Dong} and G. {Shi} and L. {Li} and X. {Li}},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Accurate and Lightweight Image Super-Resolution with Model-Guided Deep Unfolding Network}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSTSP.2020.3037516}}


## Contents
1. [Requirements](#Requirements)
2. [Test](#test)
3. [Acknowledgements](#acknowledgements)

## Requirements
- Python 3 
- skimage
- imageio
- Pytorch (Pytorch version 1.0.1 is recommended)
- tqdm 
- cv2 (pip install opencv-python)


## Test

#### Quick start

#### Test on standard SR benchmark

1. If you have cloned this repository, the pre-trained models can be found in experiment fold and test dataset Set5 can be found in data fold.

2. Then, run command:
   
   cd code_AG
   sh test.sh
   
3. Finally, PSNR values are shown on your screen, you can find the reconstruction images in `../experiment/xx/results/`. 


## Acknowledgements
- This code is built on [RCAN (PyTorch)](https://github.com/yulunzhang/RCAN). We thank the authors for sharing their codes.

