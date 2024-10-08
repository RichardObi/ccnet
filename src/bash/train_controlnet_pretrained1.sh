
echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

echo "Create datalist -> src/python/preprocessing/create_datalist.py"
output_path="project/outputs/ids"
root_path="/home/richard/Desktop/software/data/ldm_data_phase_one"
python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}



# TODO project/outputs/runs/LDM_CCNET_ANY_ACQ_TXT_v17/ldm_final_model_val_loss_06331433683314406_epoch100.pth




echo "Run train_controlnet.py -> src/python/training/train_controlnet.py"
seed=42
run_dir="aekl_v0_ldm_v0_controlnet_v0_notext_pretrained1"
# Setting of artifact urls
#stage1_uri="project/outputs/runs/aekl_v1/checkpoint_best_val_loss_e10.pth"
#stage1_uri= # TODO: Set AE_KL artifact url "/project/mlruns/837816334068618022/39336906f86c4cdc96fb6464b88c8c06/artifacts/final_model"
#ddpm_uri= # TODO: Set ldm artifact url "/project/mlruns/102676348294480761/a53f700f40184ff49f5f7e27fafece97/artifacts/final_model"
#stage1_uri="mlruns/809882896951469465/b32e98be004e4fdc92b11c546a9059e7/artifacts/final_model"
#ddpm_uri="mlruns/.../.../artifacts/final_model"
ddpm_uri="project/outputs/checkpoints/controlnet_no_text/diffusion_best_model.pth"
stage1_uri=""
# TODO Do we init_from_unet ?
training_ids="project/outputs/ids/train.tsv"
validation_ids="project/outputs/ids/validation.tsv"
config_file="configs/controlnet/controlnet_v0.yaml"
scale_factor=0.01 #=0.3
batch_size=2 #8 #16 #384 #8 #16 #64 #256 #512
n_epochs=105 #150
eval_freq=1
num_workers=4 #16 #64
experiment="CONTROLNET_pretrained1_notext_v1"
is_resumed=true #false #true
use_pretrained=1 # loading only the VAE but not the LDM as pretrained models (from source_model)
source_model="stabilityai/stable-diffusion-2-1-base"
torch_detect_anomaly=0 # whether to use torch.autograd.detect_anomaly() or not (o not, 1 yes)
#early_stopping_after_num_epochs=20
#--early_stopping_after_num_epochs ${early_stopping_after_num_epochs} \
img_width=512
img_height=512
#clip_grad_norm_by=10.0
#clip_grad_norm_or_value='value'
controlnet_conditioning_scale=1.0

if $is_resumed ; then
  python3 src/python/training/train_controlnet.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --ddpm_uri ${ddpm_uri} \
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
        --scale_factor=${scale_factor} \
        --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
        --"is_resumed" \
        --"is_ldm_fine_tuned" \
        --"use_default_report_text" \
        --"init_from_unet"
        #--clip_grad_norm_by ${clip_grad_norm_by} \
        #--clip_grad_norm_or_value ${clip_grad_norm_or_value} \
else
  python3 src/python/training/train_controlnet.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --ddpm_uri ${ddpm_uri} \
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
        --scale_factor=${scale_factor} \
        --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
        --"is_ldm_fine_tuned" \
        --"use_default_report_text" \
        --"init_from_unet"
        #--clip_grad_norm_by ${clip_grad_norm_by} \
        #--clip_grad_norm_or_value ${clip_grad_norm_or_value} \

fi
