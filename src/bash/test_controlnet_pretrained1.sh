

echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

echo "Create datalist -> src/python/preprocessing/create_datalist.py"
output_path="project/outputs/ids"
root_path="/home/richard/Desktop/software/data/ldm_data_phase_one"
python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}

echo "Run run_controlnet_inference.py -> src/python/training/run_controlnet_inference.py"
seed=42
num_inference_steps=100
y_size=64 #28 # 64 #28 #64 #20 # size of noise embedding, e.g. noisy_e shape: torch.Size([3, 4, 64, 64])
x_size=64 #28 #64 #28 #64 #28
guidance_scale=0.01 # TODO: test different text-guidance scales here
scheduler_type=ddim #ddpm #ddpm #ddim # TODO: test different schedulers here # See https://huggingface.co/docs/diffusers/en/api/schedulers/ddim#tips
controlnet_conditioning_scale=1.0 #0.8 #1.4 #1.0 # TODO: test different controlnet scales here
output_dir="project/outputs/inference/controlnet_v0_notext_pretrained1/gs0_cgs1_100timesteps/ddim/"
ddpm_uri="project/outputs/checkpoints/controlnet_no_text/diffusion_best_model.pth"
controlnet_uri="project/outputs/runs/aekl_v0_ldm_v0_controlnet_v0_notext_pretrained1/controlnet_best_model.pth" #"project/outputs/checkpoints/controlnet_no_text/controlnet_best_val_loss_00016192146068064058_epoch45.pth"
stage1_uri=""
test_ids="project/outputs/ids/test.tsv"
config_file="configs/controlnet/controlnet_v0.yaml"
batch_size=1 #8 #16 #384 #8 #16 #64 #256 #512
num_workers=4 #16 #64
experiment="controlnet_no_text_inference"
use_pretrained=1 # loading only the VAE but not the LDM as pretrained models (from source_model)
source_model="stabilityai/stable-diffusion-2-1-base"
scale_factor=0.01 #=0.3
img_width=512
img_height=512


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
      --"is_ldm_fine_tuned" \
      --"use_default_report_text" \
      --"init_from_unet" \
      --guidance_scale ${guidance_scale}
      #--"cond_on_acq_times"
      #--"is_stage1_fine_tuned"
