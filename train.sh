#!/bin/bash
python train.py --exp-name exp1 --model model1 --loss mse --KNNstep 1
python train.py --exp-name exp2 --model model1 --loss msie --KNNstep 1
python train.py --exp-name exp3 --model model1 --loss focal --KNNstep 1
python train.py --exp-name exp4 --model model1 --loss combined --KNNstep 1
python train.py --exp-name exp5 --model model1 --loss mse --KNNstep 8
python train.py --exp-name exp6 --model model1 --loss msie --KNNstep 8
python train.py --exp-name exp7 --model model1 --loss focal --KNNstep 8
python train.py --exp-name exp8 --model model1 --loss combined --KNNstep 8
