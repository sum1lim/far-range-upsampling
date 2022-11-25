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
pip install point-transformer-pytorch
```
The above package is the implementation of Point Transformer in Pytorch by the following paper. Separate installation is not necessary if `pip install -r requirements.txt` is run in the previous step.
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
