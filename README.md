### A Simple Codebase for Image-based Person Re-identification

#### Requirements: Python 3.6, Pytorch 1.6.0, yacs

#### Supported losses
##### Classification Losses
- [x] ArcFace Loss
##### Pairwise Losses
- [x] Triplet Loss

#### Supported models
- [x] ResNet-50

#### Get Started
- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default.py` with your own `data path` and `output path`, respectively.
- Run `train.sh`

#### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @InProceedings{CVPR2019IANet
    author = {Hou, Ruibing and Ma, Bingpeng and Chang, Hong and Gu, Xinqian and Shan, Shiguang and Chen, Xilin},
    title = {Interaction-And-Aggregation Network for Person Re-Identification},
    booktitle = {CVPR},
    year = {2019}
    }
