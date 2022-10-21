CUDA_VISIBLE_DEVICES=4 python inference_scoring.py \
    --tag "mscoco-new" \
    --data_folder "mscoco_split" \
    --root "/data/ljx/result/para_model/t5-base-stage_2022-04-27" \
    --checkpoint "checkpoints/checkpoint-31250" \
    --source_path "data/{}/test_new_input.txt" \
    --target_path "data/{}/test_new_target.txt" \
    --decoding "sampling" \
    --beam_size 15 \
    --model "t5-base" \
    --k 15 \
    --p 0.92 \
    --temperature 1.5 \
    --num_generate 10 \
    --seed "1234" \
    --gen_out "inference" \
    --metrics "bleu,ibleu,rouge,self-bleu" \
    --inference \
    --dir_cache "qqp-t5_base-sampling-T_2/t5-base_ft-2epoch-1000step_continued_2022-05-06/t5-base-abla-75k_2022-06-22"
    
    #--inference \
    #--scoring \