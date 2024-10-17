#! /bin/bash

#sleep 2h
#### Preliminaries

#echo "1. Activating virtual environment called generative_breast_controlnet_env."
#python3.8 -m venv MMG_env
#source ../generative_breast_controlnet_env/bin/activate
source frd_env/bin/activate

#echo "2. Pip install dependencies"
#pip3 install --upgrade pip --quiet
#pip3 install keras
#pip3 install wget --quiet
#pip3 install numpy --quiet
#pip3 install opencv-contrib-python --quiet
#pip3 install opencv-python==4.5.5.64
#pip3 install torchmetrics
#pip install numpy==1.21
#pip install torchmetrics[image]
#pip install pyradiomics
#pip install SimpleITK

#conda install -c conda-forge cudnn=7.6.5=cuda10.1_0 cudatoolkit=11.2

#conda install -c conda-forge cudatoolkit=11.2 cudnn=7.6.5=cuda10.1_0
#conda install tensorflow-gpu

#export CUDA_VISIBLE_DEVICES=0

echo "1. frd computation on TEST DATASET: Start"



#python3.8 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p1/
#python3.8 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p2/
#python3.8 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p3/
#python3.8 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p4/
#python3.8 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/PIL/

#echo "==========================================================================="
#echo "======================== FULL IMAGE ========================"
#echo "==========================================================================="

#echo "======================== REAL PRE -REAL POST Comparisons ========================"

#echo "precontrast real - postcontrast phase 1 real normalized imagenet"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B/test_B_only_p1 --resize_size 224 # --mask_dir project/outputs/inference/masks --is_mask_used  #--phase 0001 --normalize_images --limit 99999999 --model imagenet --description real_pre_vs_real_p1_imagenet_normalized_same_cases_compared_wo_split

#echo "precontrast real - postcontrast phase 2  real  normalized imagenet"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B/test_B_only_p2 --resize_size 224 # --mask_dir project/outputs/inference/masks --is_mask_used --phase 0002 --normalize_images --limit 99999999 --model imagenet --description real_pre_vs_real_p2_imagenet_normalized_same_cases_compared_wo_split

#echo "precontrast real - postcontrast phase 3 real  normalized imagenet"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B/test_B_only_p3 --resize_size 224 # --mask_dir project/outputs/inference/masks --is_mask_used --phase 0003 --normalize_images --limit 99999999 --model imagenet --description real_pre_vs_real_p3_imagenet_normalized_same_cases_compared_wo_split

#echo "precontrast real - postcontrast phase 4  real  normalized imagenet"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B/test_B_only_p4 --resize_size 224 # --mask_dir project/outputs/inference/masks --is_mask_used --phase 0004 --normalize_images --limit 99999999 --model imagenet --description real_pre_vs_real_p4_imagenet_normalized_same_cases_compared_wo_split

#echo "======================== REAL-SYNTHETIC Comparisons ========================"

#echo "postcontrast phase 1 real - postcontrast phase 1 syn normalized imagenet without masks"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p1 --resize_size 224 #--mask_dir project/outputs/inference/masks --is_mask_used  #--phase 0001 --secondphase 0001 --normalize_images --limit 99999999 --model imagenet --description real_p1_vs_syn_p1_imagenet_normalized_same_cases_compared_wo_split

#echo "postcontrast phase 2 real - postcontrast phase 2 syn normalized imagenet without masks"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p2 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p2 --resize_size 224 #--mask_dir project/outputs/inference/masks --is_mask_used #--phase 0002 --secondphase 0002 --normalize_images --limit 99999999 --model imagenet --description real_p2_vs_syn_p2_imagenet_normalized_same_cases_compared_wo_split

#echo "postcontrast phase 3 real - postcontrast phase 3 syn normalized imagenet without masks"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p3 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p3 --resize_size 224 #--mask_dir project/outputs/inference/masks --is_mask_used #--phase 0001 --secondphase 0001 --normalize_images --limit 99999999 --model imagenet --description real_p3_vs_syn_p3_imagenet_normalized_same_cases_compared_wo_split

#echo "postcontrast phase 4 real - postcontrast phase 4 syn normalized imagenet without masks"
#python3.8 src/python/eval/frd.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p4 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v0/actually_ddim/gs0_cgs1p0_200steps/p4 --resize_size 224 #--mask_dir project/outputs/inference/masks --is_mask_used #--phase 0004 --secondphase 0004 --normalize_images --limit 99999999 --model imagenet --description real_p4_vs_syn_p4_imagenet_normalized_same_cases_compared_wo_split


echo "2. frd computation on TEST DATASET: Done"

