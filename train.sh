#!/bin/bash
python train.py --exp-name exp1 --model model0_0 --loss msie --KNNstep 1 --patience 100
python train.py --exp-name exp2 --model model0_1 --loss msie --KNNstep 1
python train.py --exp-name exp3 --model model1 --loss msie --KNNstep 1
python train.py --exp-name exp4 --model model2 --loss msie --KNNstep 1
python train.py --exp-name exp5 --model model3 --loss msie --KNNstep 1
