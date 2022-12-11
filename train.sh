#!/bin/bash
python train.py --exp-name exp1 --model model0_0 --loss msie --KNNstep 1
python train.py --exp-name exp2 --model model0_1 --loss msie --KNNstep 1
python train.py --exp-name exp3 --model model1 --loss msie --KNNstep 1
python train.py --exp-name exp4 --model model2 --loss msie --KNNstep 1

python train.py --exp-name exp5 --model model1 --loss msie --KNNstep 2
python train.py --exp-name exp6 --model model1 --loss msie --KNNstep 4
python train.py --exp-name exp7 --model model1 --loss msie --KNNstep 8

python train.py --exp-name exp8 --model model1 --loss mse --KNNstep 1

python train.py --exp-name exp9 --model model1 --loss focal --focal-thresh 500 --KNNstep 1
python train.py --exp-name exp10 --model model1 --loss focal --focal-thresh 1000 --KNNstep 1
python train.py --exp-name exp11 --model model1 --loss focal --focal-thresh 2000 --KNNstep 1
python train.py --exp-name exp12 --model model1 --loss focal --focal-thresh 3000 --KNNstep 1
python train.py --exp-name exp13 --model model1 --loss focal --focal-thresh 1500 --KNNstep 1
python train.py --exp-name exp14 --model model1 --loss focal --focal-thresh 2500 --KNNstep 1

python train.py --exp-name exp15 --model model1 --loss combined --focal-thresh 2500 --focal-weight 1 --KNNstep 1
python train.py --exp-name exp16 --model model1 --loss combined --focal-thresh 2500 --focal-weight 2 --KNNstep 1
python train.py --exp-name exp17 --model model1 --loss combined --focal-thresh 2500 --focal-weight 4 --KNNstep 1
python train.py --exp-name exp18 --model model1 --loss combined --focal-thresh 2500 --focal-weight 8 --KNNstep 1