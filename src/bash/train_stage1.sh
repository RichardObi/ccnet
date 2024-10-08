

echo "Run start_script.sh -> src/bash/start_script.sh"
sh src/bash/start_script.sh

echo "Create datalist -> src/python/preprocessing/create_datalist.py"
output_path="project/outputs/ids"
root_path="/home/richard/Desktop/software/data/ldm_data_phase_one"
python3 src/python/preprocessing/create_datalist.py \
        --output_path ${output_path} \
        --root_path ${root_path}


echo "Run train_aekl.py -> src/python/training/train_aekl.py"
seed=42
run_dir="aekl_v1"
training_ids="project/outputs/ids/train.tsv"
validation_ids="project/outputs/ids/validation.tsv"
config_file="configs/stage1/aekl_v0.yaml"
batch_size=8 #64 #256
n_epochs=100
adv_start=10
eval_freq=5
num_workers=4 #16
experiment="AEKL_v1_25112023"
is_resumed=true #false #true
early_stopping_after_num_epochs=10
torch_detect_anomaly=0 # whether to use torch.autograd.detect_anomaly() or not (o not, 1 yes)
img_width=512
img_height=512
clip_grad_norm_by=10.0
clip_grad_norm_or_value='value'

if $is_resumed ; then
  python3 src/python/training/train_aekl.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --config_file ${config_file} \
        --batch_size ${batch_size} \
        --n_epochs ${n_epochs} \
        --adv_start ${adv_start} \
        --eval_freq ${eval_freq} \
        --num_workers ${num_workers} \
        --experiment ${experiment} \
        --early_stopping_after_num_epochs ${early_stopping_after_num_epochs} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \
        --"is_resumed"
else
    python3 src/python/training/train_aekl.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --config_file ${config_file} \
        --batch_size ${batch_size} \
        --n_epochs ${n_epochs} \
        --adv_start ${adv_start} \
        --eval_freq ${eval_freq} \
        --num_workers ${num_workers} \
        --early_stopping_after_num_epochs ${early_stopping_after_num_epochs} \
        --torch_detect_anomaly \
        --experiment ${experiment} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \

fi
