

echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

output_path="project/outputs/ids/"
#root_path="/home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602"
root_path="../data/ldm_data_all_phases_cropped_split1602"

echo "Create datalist based on root path ${root_path} -> src/python/preprocessing/create_datalist.py"

python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}

echo "Run train_ldm.py -> src/python/training/train_ldm.py"
seed=42
run_dir="LDM_CCNET_ANY_ACQ_v0"
stage1_uri=""
training_ids="project/outputs/ids/train.tsv"
validation_ids="project/outputs/ids/validation.tsv"
config_file="configs/ldm/ldm_v6.yaml" # TODO: Try higher learning rate e.g. 5e-5 as higher lr already improved quality (compared: v9 to v12)
scale_factor=0.1 #0.01 #0.1 #0.01 #8 #0.3
batch_size=32 #16 #16 #16 #32 #8 #16 #64 #256 #512
#--"fine_tune" \
n_epochs=100
adv_start=10
eval_freq=1
num_workers=8 #163
experiment=${run_dir}
is_resumed=true #false #true
use_pretrained=1
clip_grad_norm_by=15.0 #10.0
clip_grad_norm_or_value='value' #'value'
img_width=224 #512
img_height=224 #512
source_model="stabilityai/stable-diffusion-2-1-base" #"stabilityai/stable-diffusion-2-1-base" #"stabilityai/stable-diffusion-xl-base-1.0"   #"stabilityai/stable-diffusion-2-1-base"
torch_detect_anomaly=0 # whether to use torch.autograd.detect_anomaly() or not (0 not, 1 yes)
#early_stopping_after_num_epochs=20
#--early_stopping_after_num_epochs ${early_stopping_after_num_epochs} \

if $is_resumed ; then
  python3 src/python/training/train_ldm.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --config_file ${config_file} \
        --scale_factor=${scale_factor} \
        --batch_size ${batch_size} \
        --n_epochs ${n_epochs} \
        --eval_freq ${eval_freq} \
        --num_workers ${num_workers} \
        --experiment ${experiment} \
        --use_pretrained ${use_pretrained} \
        --source_model ${source_model} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --img_width ${img_width} \
        --img_height ${img_height} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \
        --"cond_on_acq_times" \
        --"use_default_report_text" \
        --"is_resumed"
        #--"fine_tune" \


else
  python3 src/python/training/train_ldm.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --config_file ${config_file} \
        --scale_factor=${scale_factor} \
        --batch_size ${batch_size} \
        --n_epochs ${n_epochs} \
        --eval_freq ${eval_freq} \
        --num_workers ${num_workers} \
        --experiment ${experiment} \
        --use_pretrained ${use_pretrained} \
        --source_model ${source_model} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --img_width ${img_width} \
        --img_height ${img_height} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \
        --"cond_on_acq_times" \
        --"use_default_report_text" \
        #--"fine_tune" \


fi
