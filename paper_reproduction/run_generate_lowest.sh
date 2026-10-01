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
                    for temper in 0.0
                    do
                        for top_p in 1.0
                        do
                            for name in member nonmember
                            do
                                float_lrate=$(awk "BEGIN {print $lrate}")
                                python generate_lowest_ft_more_layers.py --model $model_name --epoch $epoch_num --size $split_size --subname $data_name --lr $float_lrate --temp $temper --topp $top_p --candidate $name
                            done
                        done
                    done
                done
            done
        done
    done
done
