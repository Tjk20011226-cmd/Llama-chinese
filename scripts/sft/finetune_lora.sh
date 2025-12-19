output_model=ft_model
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
export CUDA_HOME=/usr/local/cuda-12.1/
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export USE_SYSTEM_NCCL=1
cp train/sft/finetune.sh ${output_model}
deepspeed --include localhost:0 train/sft/finetune_clm_lora.py \
    --model_name_or_path my_model/model_1 \
    --train_files data/微调数据/random_cl.csv \
    --validation_files  data/微调数据/val_ccpm_cl.csv \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 20 \
    --eval_steps 20 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    --resume_from_checkpoint my_model/model_1 \
    | tee -a ${output_model}/train.log
    


    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
