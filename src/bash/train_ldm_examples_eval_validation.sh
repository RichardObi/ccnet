

echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

echo "Create datalist -> src/python/preprocessing/create_datalist.py"
output_path="project/outputs/ids"
root_path="/home/richard/Desktop/software/data/ldm_data_phase_one"
python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}


echo "Run train_ldm.py -> src/python/training/train_ldm.py"
seed=42
run_dir="aekl_v1_ldm_v0"
#stage1_uri="project/outputs/runs/aekl_v1/checkpoint_best_val_loss_e10.pth"
# TODO: Set AE_KL artifact url "/project/mlruns/837816334068618022/39336906f86c4cdc96fb6464b88c8c06/artifacts/final_model"
stage1_uri="mlruns/809882896951469465/b32e98be004e4fdc92b11c546a9059e7/artifacts/final_model"
training_ids="project/outputs/ids/train.tsv"
validation_ids="project/outputs/ids/validation.tsv"
config_file="configs/ldm/ldm_v0.yaml"
scale_factor=0.01 #=0.3
batch_size=1 #16 #64 #256 #512
n_epochs=150
eval_freq=5
num_workers=2 #16
experiment="LDM_v1_03012024"
is_resumed=false #false #true
use_pretrained=1
source_model="stabilityai/stable-diffusion-2-1-base"
torch_detect_anomaly=0 # whether to use torch.autograd.detect_anomaly() or not (o not, 1 yes)
#early_stopping_after_num_epochs=20
#--early_stopping_after_num_epochs ${early_stopping_after_num_epochs} \
validation_ids="project/outputs/ids/examples/validation.tsv"
img_width=512
img_height=512
clip_grad_norm_by=10.0
clip_grad_norm_or_value='value'

if $is_resumed ; then
  python3 src/python/training/train_ldm.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --config_file ${config_file} \
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
        --"is_resumed"
        #--scale_factor=${scale_factor} \
else
  python3 src/python/training/train_ldm.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --config_file ${config_file} \
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
        #--scale_factor=${scale_factor} \

fi