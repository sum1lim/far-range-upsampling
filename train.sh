#!/bin/bash
python train.py --exp-name exp1 --model model0_1 --loss msie --KNNstep 1
python train.py --exp-name exp2 --model model0_2 --loss msie --KNNstep 1
python train.py --exp-name exp3 --model model1_1 --loss msie --KNNstep 1
python train.py --exp-name exp4 --model model1_2 --loss msie --KNNstep 1
python train.py --exp-name exp5 --model model2_1 --loss msie --KNNstep 1
python train.py --exp-name exp6 --model model2_2 --loss msie --KNNstep 1