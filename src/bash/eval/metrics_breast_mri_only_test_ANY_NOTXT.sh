#! /bin/bash

#sleep 2h

#### Preliminaries


#echo "1. Activating virtual environment called generative_breast_controlnet_env."
##python3 -m venv MMG_env
#source ../generative_breast_controlnet_env/bin/activate

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

#git clone https://github.com/Project-MONAI/GenerativeModels.git
#cd GenerativeModels/
#python setup.py install
#cd ..


#python3 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p1
#python3 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p2
#python3 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p3
#python3 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p4
#python3 src/python/eval/flip_and_rotate.py --dataset_path_1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/PIL

echo "3. METRIC computation on TEST data."
echo "==========================================================================="
echo "======================== FULL IMAGE ========================"
echo "==========================================================================="

echo "======================== REAL PRE -REAL POST Comparisons ========================"

#echo "precontrast real - postcontrast phase 1 real  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p1 --phase 0001 --normalize_images

#echo "precontrast real - postcontrast phase 2  real  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p2  --phase 0002 --normalize_images

#echo "precontrast real - postcontrast phase 3 real  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602//test/test_B_only_p3 --phase 0003 --normalize_images

#echo "precontrast real - postcontrast phase 4  real  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_A /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602//test/test_B_only_p4 --phase 0004 --normalize_images


echo "======================== REAL-SYNTHETIC Comparisons ========================"

#echo "postcontrast phase 1 real - postcontrast phase 1 synthetic normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p1 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p1 --phase 0001 --secondphase 0001 --normalize_images

#echo "postcontrast phase 2 real - postcontrast phase 2 synthetic  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p2 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p2 --phase 0002 --secondphase 0002 --normalize_images

#echo "postcontrast phase 3 real - postcontrast phase 3 synthetic  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p3 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p3 --phase 0003 --secondphase 0003 --normalize_images

#echo "postcontrast phase 4 real - postcontrast phase 4 synthetic  normalized"
#python3 src/python/eval/metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602/test/test_B_only_p4 /home/richard/Desktop/software/generative_breast_controlnet/project/outputs/inference/INF_CTRL_CCNET_ANY_ACQ_v1/actually_ddim/gs0_cgs1p2_200steps/p4 --phase 0004 --secondphase 0004 --normalize_images


echo "FINISHED CALCULATING METRICS"