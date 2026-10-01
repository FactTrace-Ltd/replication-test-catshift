#!/bin/sh


for model_name in 410m
do
    for epoch_num in 9
    do
        for data_name in europarl
        
        do
            for split_size in 600
            do
                for lrate in 8e-5
                do
                    float_lrate=$(awk "BEGIN {print $lrate}")
                    python main_pile_subset_saved_model_pythia.py --model $model_name --epoch $epoch_num --size $split_size --subname $data_name --lr $float_lrate
                done
            done
        done
    done
done