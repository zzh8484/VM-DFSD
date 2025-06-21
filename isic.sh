#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"

datasets=(
    "isic2018"
    "isic2017"
)




for dataset in "${datasets[@]}"
do
    python isic_mytrain.py --name "base" --dataset "$dataset" --scan 3 --group 8 --epoch 300 --lr 0.001
    
    if [ $? -eq 0 ]; then
        echo "数据集 $dataset 运行脚本成功。"
    else
        echo "数据集 $dataset 运行脚本失败。"
    fi
done
