#!/bin/sh


for data_name in europarl
do
    for e in 9
    do
        for log in mlp
        do
            python sum_norm_loss_pvalue.py --subname $data_name --epoch $e --logging $log
        done
    done
done
