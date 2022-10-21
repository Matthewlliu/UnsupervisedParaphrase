CUDA_VISIBLE_DEVICES=4 python t5gen.py \
    --start 130000 \
    --step 5000 \
    --test_id "0804" \
    --sem_weight 4.5 \
    --ppl_weight 8.0 \
    --div_weight 0.7 \
    --kw_drop_per 0.09 \
    --kw_sub_per 1.0 \
    --kw_sub_temp 1.5 \
    --temperature 1.0 \
    --sem_model /data/ljx/cpt/roberta-large-mnli \
    --if_save
    #--sem_model /data/MODELS/t5-large #finetuning-t5/mnli/cache/t5-base_best
    
