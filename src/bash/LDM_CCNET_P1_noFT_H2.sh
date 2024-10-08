

echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

echo "Create datalist -> src/python/preprocessing/create_datalist.py"
output_path="project/outputs/ids/p1"
#root_path="/home/richard/Desktop/software/data/ldm_data_phase_one_split1602"
root_path="../data/ldm_data_phase_one_split1602"

python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}

echo "Run train_ldm.py -> src/python/training/train_ldm.py"
seed=42
run_dir="LDM_CCNET_P1_noFT_H2_v0" #"LDM_CCNET_P1_ACQ_TXT_v6" changed on 23.02.2024
stage1_uri=""
training_ids="project/outputs/ids/p1/train.tsv"
validation_ids="project/outputs/ids/p1/validation.tsv"
config_file="configs/ldm/ldm_v0.yaml"
scale_factor=0.01 #0.01 #8 #0.3
batch_size=4 #32 #8 #16 #64 #256 #512
n_epochs=100
adv_start=10
eval_freq=1
num_workers=4 #163
experiment=${run_dir}
is_resumed=true #false #true
use_pretrained=1
clip_grad_norm_by=1.0
#--"cond_on_acq_times"
clip_grad_norm_or_value='value'
img_width=512
img_height=512
source_model="stabilityai/stable-diffusion-2-1-base" #"stabilityai/stable-diffusion-2-1-base" #"stabilityai/stable-diffusion-xl-base-1.0"   #"stabilityai/stable-diffusion-2-1-base"
torch_detect_anomaly=0 # whether to use torch.autograd.detect_anomaly() or not (0 not, 1 yes)
#early_stopping_after_num_epochs=20
#--early_stopping_after_num_epochs ${early_stopping_after_num_epochs} \
lr_warmup_steps=500
snr_gamma=5.0

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
        --img_width ${img_width} \
        --img_height ${img_height} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \
        --snr_gamma ${snr_gamma} \
        --lr_warmup_steps ${lr_warmup_steps} \
        --"use_default_report_text" \
        --"is_resumed" \
        #--"cond_on_acq_times"

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
        --img_width ${img_width} \
        --img_height ${img_height} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \
        --snr_gamma ${snr_gamma} \
        --lr_warmup_steps ${lr_warmup_steps} \
        --"use_default_report_text" \
        #--"cond_on_acq_times"

fi