
echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

echo "Create datalist -> src/python/preprocessing/create_datalist.py"

output_path="project/outputs/ids/any"
#root_path="/home/richard/Desktop/software/data/ldm_data_all_phases_cropped_split1602"
root_path="../data/ldm_data_all_phases_cropped_split1602"
python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}


echo "Run run_controlnet_inference.py -> src/python/training/run_controlnet_inference.py"
seed=42
num_inference_steps=200 #1000 #200
y_size=28 # 64 #28 # 64 #28 #64 #20 # size of noise embedding, e.g. noisy_e shape: torch.Size([3, 4, 64, 64])
x_size=28 # 64 #28 #64 #28 #64 #28
experiment="INF_CTRL_CCNET_ANY_ACQ_TXT_v1"
guidance_scale=1.0 #.0 #0.01 # TODO: test different text-guidance scales here
scheduler_type=ddim #ddpm #ddpm #ddpm #ddim # TODO: test different schedulers here # See https://huggingface.co/docs/diffusers/en/api/schedulers/ddim#tips
controlnet_conditioning_scale=1.1 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p1_200steps_e99/"
ddpm_uri="project/outputs/runs/LDM_CCNET_ANY_ACQ_TXT_v17/ldm_final_model_val_loss_06331433683314406_epoch100.pth"
controlnet_uri="project/outputs/runs/CTRL_CCNET_ANY_ACQ_TXT_v0/controlnet_best_val_loss_007992724368850848_epoch99.pth"   #controlnet_best_val_loss_008037116141761028_epoch49.pth" #controlnet_best_val_loss_007992724368850848_epoch99.pth"
stage1_uri=""
test_ids="project/outputs/ids/any/test.tsv"
config_file="configs/controlnet/controlnet_v1.yaml"
scale_factor=0.1 #0.18215 #0.1  #0.01 #=0.3
batch_size=1 #8 #16 #384 #8 #16 #64 #256 #512
num_workers=4 #16 #64
use_pretrained=1 # loading only the VAE but not the LDM as pretrained models (from source_model)
source_model="stabilityai/stable-diffusion-2-1-base"
img_width=224 #512
img_height=224 #512

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"


controlnet_conditioning_scale=0.9 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs0p9_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.2 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p2_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.3 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p3_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.4 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p4_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.5 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p5_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.7 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p7_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.6 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p6_200steps_e99/"

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      --"cond_on_acq_times" \
      #--"only_phase_1"
      #--"use_default_report_text"
      #--upper_limit ${upper_limit} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"

controlnet_conditioning_scale=1.8 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/${experiment}/actually_ddim/gs1_cgs1p8_200steps_e99/"

