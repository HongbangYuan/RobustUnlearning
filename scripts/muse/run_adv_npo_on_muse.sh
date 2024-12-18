#!/bin/bash
# Docker Environment:  210.75.240.150:30003/hongbang/personal:ubuntu_focal_conda_llama-factory
# 设置huggingface缓存目录环境变量，这样就可以提前下载好huggingface的模型到/netcache/huggingface。
export HF_HOME=/mnt/publiccache/huggingface
pip install rouge
pip install transformers==4.37.2
pip install evaluate
pip install natsort
pip install trl>=0.8.1
# 设置huggingface离线模式环境变量，这样使得程序完全依靠本地缓存，不尝试连接huggingface
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /mnt/userdata/projects/LLMUnlearn/LLaMA_Factory || exit
cp -r /mnt/userdata/nltk_data /root
ls -l /root/nltk_data

#*************************utilities****************************

# Function to check if a GPU is occupied
check_gpu_occupied() {
    local gpu=$1
    if [[ $(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader | wc -l) -gt 0 ]]; then
        return 0  # GPU is occupied
    else
        return 1  # GPU is not occupied
    fi
}
get_free_gpu() {
    while true; do
        for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
            if [[ $(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader | wc -l) -eq 0 ]]; then
                echo $gpu
                return
            fi
        done
        sleep 1
    done
}
run_evaluation() {
    local checkpoint=$1
    local gpu=$2
    local port=$3
    local checkpoint_num=$(basename "$checkpoint" | sed 's/checkpoint-//')
    local eval_path=/mnt/userdata/projects/muse_bench
    echo "Processing ${checkpoint} on GPU ${gpu} with port ${port} in directory ${eval_path}"

    cd ${eval_path} || exit
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=./ python eval.py  --model_dirs "${checkpoint}" \
      --names "${checkpoint_num}" --corpus news --tokenizer_dir /mnt/publiccache/hongbang/Llama-2-7b-chat-hf \
      --out_file "${real_output_dir}/checkpoint-${checkpoint_num}/result.csv"  &
    pids+=($!)  # Save the process ID
    cd /mnt/userdata/projects/LLMUnlearn || exit

    # Ensure the GPU is occupied
    local timeout=256
    local interval=1
    local elapsed=0

    while ! check_gpu_occupied $gpu; do
        sleep $interval
        elapsed=$((elapsed + interval))
        if [[ $elapsed -ge $timeout ]]; then
            echo "Error: GPU ${gpu} not occupied after ${timeout} seconds"
            kill $pid
            pids=(${pids[@]/$pid})  # Remove the PID from the list
            return 1  # Indicate failure
        fi
    done
    return 0  # Indicate success
}

#************************* End of Utilities ****************************


model_path="/model/MUSE-news_target"
src_path='/mnt/publiccache/hongbang/MUSE-news_target'
echo "Copying model from ${src_path} to ${model_path}"
mkdir -p ${model_path}
cp -r ${src_path}/* ${model_path}

save_path="/saves"
mkdir -p ${save_path}
echo "Create save path $save_path"

#loss_types=("npo" "npo_grad_diff" "npo_KL")
#loss_types=("npo_grad_diff" "npo_KL")
loss_type="npo"
#loss_type="npo_grad_diff"

lr=3e-6
#lrs=(3e-6 5e-6 7e-6 9e-6)
#npo_coeffs=(0.2 0.4 0.6 0.8)
npo_coeffs=(1)
#for loss_type in "${loss_types[@]}"
#for lr in "${lrs[@]}"
for npo_coeff in "${npo_coeffs[@]}"
do
  echo "----------------------loss type: ${loss_type}-------------------------------"
    batch_size=8
    gradient_accumulation_steps=1
    num_epochs=10

    tag=${loss_type}_epoch${num_epochs}_batchsize${batch_size}_gradient_accumulation_steps${gradient_accumulation_steps}_lr${lr}_with_adv
    echo "-------------Experiment Tag:${tag}-------------------------"
    output_dir=${save_path}/MUSE/${tag}
    real_output_dir="/mnt/userdata/projects/LLMUnlearn/LLaMA_Factory/saves/MUSE/run_in_groups/${loss_type}/${tag}"
    echo "Results will be save in ${output_dir}"
    echo "Loss Type: ${loss_type}"

    log_file=run_${tag}.log
    mkdir -p ./logs/MUSE
    echo "Log file ${log_file}"

    PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage adv_npo --npo_loss ${loss_type} --npo_coeff ${npo_coeff}  \
    --adv_npo_perturb_layer 4 --adv_npo_epsilon 8 \
    --model_name_or_path ${model_path} --do_train  --save_strategy epoch --save_only_model \
    --dataset MUSE_forget_and_retain1 --dataset_dir ./data --finetuning_type full \
    --output_dir ${output_dir} --overwrite_cache \
    --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size 8 --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 \
    --eval_steps 30000 --evaluation_strategy steps  --template llama2 \
    --learning_rate ${lr} --num_train_epochs ${num_epochs} --val_size 0.0000001 --plot_loss  \
    --dataset_name muse \
    --fp16 --eval_dataset_dir ./data/MUSE/  2>&1 | tee ./logs/MUSE/${log_file}

    cd /mnt/userdata/projects/LLMUnlearn || exit

    #**********************************Running Evaluation*******************************

    mkdir -p ${real_output_dir}
    echo "Create real save path ${real_output_dir}"
    checkpoints=$(find "${output_dir}" -type d -name "checkpoint-*")
    echo "Detected checkpoints: ${checkpoints}"

    pids=()  # Array to hold process IDs
    base_port=25640  # Starting port number
    for checkpoint in ${checkpoints}; do
        gpu=$(get_free_gpu)
        port=$((base_port + RANDOM % 10000))  # Randomize port number within a range
        run_evaluation "$checkpoint" "$gpu" "$port"
    done
    #************************Finished Evaluation********************

    # Wait for all background processes to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "All evaluations are complete."

    # Removing checkpoints and copying output directory
    for checkpoint in ${checkpoints}; do
        echo "Removing checkpoint dir: ${checkpoint}"
        rm -r ${checkpoint}
    done
    echo "Copying from ${output_dir} to ${real_output_dir}"
    cp -r ${output_dir} ${real_output_dir}
    cd /mnt/userdata/projects/LLMUnlearn/LLaMA_Factory || exit

done

echo "Finished Running!"


