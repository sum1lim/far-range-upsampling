# Far Range Upsampling
Point cloud enhancement in the far range from the ego vehicle for autonomous driving

## Installation
### Create virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Install Point Transformer package
```
git clone https://github.com/lucidrains/point-transformer-pytorch.git
cd point-transformer-pytorch
pip install -e .
cd ..
```
The above package is the implementation of Point Transformer in Pytorch by the following paper
```
@misc{zhao2020point,
    title={Point Transformer}, 
    author={Hengshuang Zhao and Li Jiang and Jiaya Jia and Philip Torr and Vladlen Koltun},
    year={2020},
    eprint={2012.09164},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
### Download Normalized Loss Functions package
```
git clone https://github.com/HanxunH/Active-Passive-Losses.git
```
The above package is the implementation of Normalized Loss Functions in Pytorch by the the following paper
```
@inproceedings{ma2020normalized,
  title={Normalized Loss Functions for Deep Learning with Noisy Labels},
  author={Ma, Xingjun and Huang, Hanxun and Wang, Yisen and Romano, Simone and Erfani, Sarah and Bailey, James},
  booktitle={ICML},
  year={2020}
}
```
