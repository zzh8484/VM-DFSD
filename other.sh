datasets=(
    "ThyroidNodule-DDTI"
    "ThyroidNodule-TG3K"
    "ThyroidNodule-TN3K"
    "Echocardiography-HMCQU"
    "Breast-BUSI"
    "Breast-UDIAT"
)


for dataset in "${datasets[@]}"
do


    echo "正在使用数据集 $dataset 运行脚本..."
    python other_mytrain.py --dataset "$dataset" --name "light224" --epoch 300 --img_size 224
    
    if [ $? -eq 0 ]; then
        echo "使用数据集 $dataset 运行脚本成功。"
    else
        echo "使用数据集 $dataset 运行脚本失败。"
    fi

done