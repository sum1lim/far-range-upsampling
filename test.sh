#!/bin/bash
test_models --model model0_0 --params ./checkpoints/exp1_model.pt --KNNstep 1
test_models --model model0_1 --params ./checkpoints/exp2_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp3_model.pt --KNNstep 1
test_models --model model2 --params ./checkpoints/exp4_model.pt --KNNstep 1

test_models --model model1 --params ./checkpoints/exp5_model.pt --KNNstep 2
test_models --model model1 --params ./checkpoints/exp6_model.pt --KNNstep 4
test_models --model model1 --params ./checkpoints/exp7_model.pt --KNNstep 8

test_models --model model1 --params ./checkpoints/exp8_model.pt --KNNstep 1

test_models --model model1 --params ./checkpoints/exp9_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp10_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp11_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp12_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp13_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp14_model.pt --KNNstep 1

test_models --model model1 --params ./checkpoints/exp15_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp16_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp17_model.pt --KNNstep 1
test_models --model model1 --params ./checkpoints/exp18_model.pt --KNNstep 1