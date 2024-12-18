#!/bin/bash
export HF_HOME=/mnt/publiccache/huggingface
pip install rouge
pip install transformers==4.44.0
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Remember to change the path according to your local directory
cd /mnt/userdata/projects/LLMUnlearn/LLaMA_Factory || exit
model_path="/model/Meta-Llama-3.1-8B-Instruct"
cp -r /mnt/userdata/nltk_data /root
ls -l /root/nltk_data


names=(
'1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera'
'6_Cindy_Crawford' '7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex'
)
learning_rates=(1e-6)
loss_types=("npo" "npo_grad_diff" "npo_KL")


echo "Running adv npo"
for lr in "${learning_rates[@]}"
do
  echo "----------------------learning rate: ${lr}-------------------------------"
  for loss_type in "${loss_types[@]}"
    do
    echo "------------------Loss Type:${loss_type}----------------------------"
    for name in "${names[@]}"
      do
        id=$name
        echo "------------${id}---------------"
        output_dir="./saves/RWKU/Meta-Llama-3.1-8B-Instruct/AdvNPO_loss_type_${loss_type}/Target/${id}"
        mkdir -p ${output_dir}
        log_file="${output_dir}/log_file.txt"a
        echo "outputdir:${output_dir}"
        echo "log file:${log_file}"

        PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage adv_npo \
        --adv_npo_perturb_layer 4 --adv_npo_epsilon 8 --npo_loss ${loss_type} \
        --model_name_or_path ${model_path} --do_train \
        --dataset ${id}_Positive_wRetain --dataset_dir ./data --finetuning_type full \
        --output_dir ${output_dir} --overwrite_cache \
        --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
        --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 \
        --eval_steps 30000 --evaluation_strategy steps --load_best_model_at_end --template llama3 \
        --learning_rate ${lr} --num_train_epochs 1.0 --val_size 0.0000001 --plot_loss \
        --output_result_dir ${output_dir} \
        --fp16 --eval_dataset_dir ./data/RWKU/Target/ \
        --target ${id} 2>&1 | tee ${log_file}
    done
  done
done



echo "Running npo"
for lr in "${learning_rates[@]}"
do
  echo "----------------------learning rate: ${lr}-------------------------------"
  for loss_type in "${loss_types[@]}"
    do
    echo "------------------Loss Type:${loss_type}----------------------------"
    for name in "${names[@]}"
      do
        id=$name
        echo "------------${id}---------------"
        output_dir="./saves/RWKU/Meta-Llama-3.1-8B-Instruct/NPO_loss_type_${loss_type}/Target/${id}"
        mkdir -p ${output_dir}
        log_file="${output_dir}/log_file.txt"
        echo "outputdir:${output_dir}"
        echo "log file:${log_file}"


        PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage npo  --dpo_beta 0.2 \
        --model_name_or_path ${model_path} --do_train --npo_loss ${loss_type}  \
        --dataset ${id}_Positive_wRetain --dataset_dir ./data --finetuning_type full \
        --output_dir ${output_dir} --overwrite_cache \
        --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
        --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 \
        --eval_steps 30000 --evaluation_strategy steps --load_best_model_at_end --template llama3 \
        --learning_rate ${lr} --num_train_epochs 1.0 --val_size 0.0000001 --plot_loss \
        --output_result_dir ${output_dir} \
        --fp16 --eval_dataset_dir ./data/RWKU/Target/ \
        --target ${id} 2>&1 | tee ${log_file}
      done
  done
done


echo "Finished Running!"