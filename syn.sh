#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"  
python syn_mytrain.py --name "our_base_1-82" --scan 3 --group 8  --epoch 300
# python syn_mytrain.py --name "our_base_1-12" --scan 5 --group 12 --epoch 300
# python syn_mytrain.py --name "our_base_9-12" --scan 4 --group 4 --epoch 300
# python syn_mytrain.py --name "our_base_5-8" --scan 6 --group 4 --epoch 300
# python syn_mytrain.py --name "our_base_1-4,9-12" --scan 7 --group 8  --epoch 300
# python syn_mytrain.py --name "our_base_5-12" --scan 8 --group 8  --epoch 300
# python syn_mytrain.py --name "our_base_1-4" --scan 0 --group 4 --epoch 300

# python syn_mytrain.py --name "our_base_nopre" --scan 3 --group 8 --pretain "False" --epoch 300

