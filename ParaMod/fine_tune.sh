#GPU_ID=$1
#TAG=$1
LENGTH=512
BATCH_SIZE=4  # batch size per each GPU
LR="1e-5"
ACCUM=4
N_EPOCHS=2
#SAVE_STEPS=62  # Validating and save checkpoints
RANDOM_SEED=1234
STL=1
TAG="kqapro-100"

#TRAIN_DATA='/home/ljx/Megatron-LM-main/tests/new_t5/output/new_bookcorpus/run_311'
TRAIN_DATA='data/kqapro_split'
#CKPT='/data/ljx/result/para_model/t5-base-stage-5epochs_2022-06-04/checkpoints/checkpoint-18750/' # 3 epoch
CKPT='/data/MODELS/t5-base'
OUT='/data/ljx/result/para_model/{}/checkpoints/'
#DEV_DATA="./data/QQP_split/dev_preprocessed.txt"

#deepspeed --include localhost:0,5 --master_port 61000 \
#main.py \

CUDA_VISIBLE_DEVICES=3 python main.py \
    --train_dir ${TRAIN_DATA}\
    --checkpoint ${CKPT} \
    --output_dir ${OUT} \
    --max_length ${LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUM} \
    --num_train_epochs ${N_EPOCHS} \
    --learning_rate ${LR} \
    --save_total_limit ${STL} \
    --tag ${TAG} \
    --model t5-base \
    --seed ${RANDOM_SEED} \
    --finetune \
    --sample_num 100 \
    #--fp16 \
    #--deepspeed deepspeed/stage3.json
    #1>>./result/${TAG}.txt 2>&1
    #--debug --toy
    #--dev_data_path ${DEV_DATA} \
    #--save_steps ${SAVE_STEPS} \