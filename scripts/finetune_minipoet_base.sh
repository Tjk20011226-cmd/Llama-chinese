#!/bin/bash

# 输出目录：MiniPoet-Base
output_model=MiniPoet-Base

if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi

cp ./finetune_clm.py ${output_model}

export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

deepspeed --include localhost:0,1,2,3 finetune_clm.py \
    --model_name_or_path 全微调_ccpm+诗词生成_2epoch \
    --train_files data/微调数据/诗词生成_简体_微调样式.csv \
                  data/微调数据/ccpm_instruction_factory.csv \
                  data/微调数据/模型生成.csv \
                  data/微调数据/诗词翻译_简体.csv \
                  data/微调数据/ccpc_关键词生成诗词_简体_微调样式.csv \
                  data/微调数据/ccpc_提取关键词_简体_微调样式.csv \
                  data/微调数据/COIG-CQIA-full.csv \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --warmup_steps 400 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 1000 \
    --eval_steps 200000 \
    --save_total_limit 5 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 512 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
