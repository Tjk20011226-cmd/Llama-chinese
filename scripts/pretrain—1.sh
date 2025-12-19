output_model=output_model
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./pretrain.sh ${output_model}
cp ./ds_config_zero*.json ${output_model}
export CUDA_HOME=/usr/local/cuda/
export NCCL_P2P_DISABLE=1

deepspeed --include localhost:0 pretrain_clm.py \
    --model_name_or_path  output_model/checkpoint-58000\
    --tokenizer_name output_model/checkpoint基座  \
    --train_files   data/训练数据/train_all_data_random.csv \
                    data/训练数据/train_news_chunk15.csv \
                    data/训练数据/train_news_chunk4.csv \
                    data/训练数据/train_my_baike_qa.csv \
                    data/训练数据/train_news_chunk16.csv \
                    data/训练数据/train_news_chunk5.csv \
                    data/训练数据/train_my_belll_3M_cn.csv \
                    data/训练数据/train_news_chunk17.csv \
                    data/训练数据/train_news_chunk6.csv \
                    data/训练数据/train_my_tran_poetry_zh_no_dulpticates.csv \
                    data/训练数据/train_news_chunk18.csv \
                    data/训练数据/train_news_chunk7.csv \
                    data/训练数据/train_my_web_text_zh.csv \
                    data/训练数据/train_news_chunk19.csv \
                    data/训练数据/train_news_chunk8.csv \
                    data/训练数据/train_news_chunk1.csv \
                    data/训练数据/train_news_chunk2.csv \
                    data/训练数据/train_news_chunk9.csv \
                    data/训练数据/train_news_chunk10.csv \
                    data/训练数据/train_news_chunk20.csv \
                    data/训练数据/train_sft_train.csv \
                    data/训练数据/train_news_chunk11.csv \
                    data/训练数据/train_news_chunk21.csv \
                    data/训练数据/train_news_chunk12.csv \
                    data/训练数据/train_news_chunk22.csv \
                    data/训练数据/train_zhihu_kol.csv \
                    data/训练数据/train_news_chunk13.csv \
                    data/训练数据/train_news_chunk23.csv \
                    data/训练数据/train_news_chunk14.csv \
                    data/训练数据/train_news_chunk3.csv \
                    data/训练数据/train_poet_data.csv\
    --validation_files  data/dev_sft.csv \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --use_fast_tokenizer false \
    --max_eval_samples 500 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --warmup_steps 5000 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 2000 \
    --eval_steps 5000000 \
    --save_total_limit 3 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --run_name ${output_model} \
    --bf16 \
    --bf16_full_eval \
    --gradient_checkpointing \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    --deepspeed ./ds_config_zero3.json \
    --resume_from_checkpoint ${output_model}/checkpoint-58000\
    | tee -a ${output_model}/train.log
