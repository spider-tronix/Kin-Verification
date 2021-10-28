#!/bin/sh
archs = ('resnet50', 'densenet161', 'vgg16')
dts = ('imagenet', 'vggface2', 'scratch')

for dt in "${dts[@]}"
do
    if [[ $dt == 'vggface2' ]]
    
    then
        python3 main.py -wd kaggleFIW -wts ../resnet50_scratch_weight.pkl -testfn test_labels.csv -trainfn train_labels.csv --arch_type resnet50_scratch
        echo "$dt resnet is completed"
        python3 main.py -wd kaggleFIW -wts ../senet50_scratch_weight.pkl -testfn test_labels.csv -trainfn train_labels.csv --arch_type senet50_scratch
        echo "$dt senet is completed"
    else
        for arch in "${archs[@]}"
        do
            python3 main.py  -wd kaggleFIW -testfn test_labels.csv -trainfn train_labels.csv --arch_type $arch -dt $dt
            echo "$dt $arch is completed"
        done
    fi
done