#!/bin/bash
export HF_HOME=/mnt/publiccache/huggingface
pip install rouge
pip install transformers==4.37.2
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Remember to change the path according to your local directory
cd /mnt/userdata/projects/LLMUnlearn/LLaMA_Factory || exit
cp -r /mnt/userdata/nltk_data /root
ls -l /root/nltk_data
model_path="/model/Meta-Llama-3-8B-Instruct"

names=(
'1_Stephen_King'
'2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera'
'6_Cindy_Crawford'
'7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex'

)
#learning_rates=(1e-7 5e-7 1e-6 5e-6)
#learning_rates=(5e-6)
lr=5e-6
epsilons=(2 4)
#epsilons=(4)
#perturb_layers=(6 5 4)
perturb_layers=(6)
#loss_types=("ga" "ga_grad_diff" "ga_KL")
#loss_types=("ga_KL")

#loss_types=("ga" "ga_grad_diff")
#loss_types=("ga_grad_diff")

echo "Running adv rmu"
for epsilon in "${epsilons[@]}"
do
  echo "----------------------epsilon:${epsilon}-------------------------------"
  for perturb_layer in "${perturb_layers[@]}"
    do

    echo "------------------perturb_layer:${perturb_layer}----------------------------"
    for name in "${names[@]}"
      do
        id=$name
        echo "------------${id}---------------"
        output_dir="./saves/RWKU/llama3_8b_instruct/adv_rmu/search_parameters/epsilon_${epsilon}_perturb_layer_${perturb_layer}/Target/${id}"
        mkdir -p ${output_dir}
        log_file="${output_dir}/log_file.txt"
        echo "outputdir:${output_dir}"
        echo "log file:${log_file}"

        PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage adv_rmu \
        --rmu_layer 7  --rmu_steering_coeff 300  --rmu_alpha 300   \
        --adv_npo_epsilon ${epsilon} --adv_npo_steps 3 --adv_rmu_perturb_layer ${perturb_layer} \
        --model_name_or_path ${model_path} --do_train    \
        --dataset ${id}_Positive_wRetain --dataset_dir ./data --finetuning_type full \
        --output_dir ${output_dir} --overwrite_cache \
        --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
        --per_device_train_batch_size 2 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 --save_strategy "no" \
        --eval_steps 30000 --evaluation_strategy "no" --load_best_model_at_end --template llama3 \
        --learning_rate ${lr} --num_train_epochs 1.0 --val_size 0.0000001 --plot_loss \
        --output_result_dir ${output_dir} \
        --fp16 --eval_dataset_dir ./data/RWKU/Target/ \
        --target ${id} 2>&1 | tee ${log_file}


      done
  done
done


echo "Running rmu"

for name in "${names[@]}"
  do
    id=$name
    echo "------------${id}---------------"
    output_dir="./saves/rmu_related/llama3_8b_instruct/Target/${id}"
    mkdir -p ${output_dir}
    log_file="${output_dir}/log_file.txt"
    echo "outputdir:${output_dir}"
    echo "log file:${log_file}"

    PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage rmu \
    --rmu_layer 7  --rmu_steering_coeff 300  --rmu_alpha 300   \
    --model_name_or_path ${model_path} --do_train    \
    --dataset ${id}_Positive_wRetain --dataset_dir ./data --finetuning_type full \
    --output_dir ${output_dir} --overwrite_cache \
    --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 \
    --eval_steps 30000 --evaluation_strategy steps --load_best_model_at_end --template llama3 \
    --learning_rate ${lr} --num_train_epochs 1.0 --val_size 0.0000001 --plot_loss \
    --output_result_dir ${output_dir} \
    --fp16 --eval_dataset_dir ./data/RWKU/Target/ \
    --target ${id} 2>&1 | tee ${log_file}


  done



echo "Finished Running!"