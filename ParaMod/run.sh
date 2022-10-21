#GPU_ID=$1
#TAG=$1
LENGTH=512
BATCH_SIZE=4  # batch size per each GPU
LR="1e-4"
ACCUM=4
N_EPOCHS=5
SAVE_STEPS=6250  # Validating and save checkpoints
RANDOM_SEED=1234
STL=1
TAG="only_test"

TRAIN_DATA='data/run_311/75k'
OUTPUT_DIR='/data/ljx/result/para_model/{}/checkpoints/'
#DEV_DATA="./data/QQP_split/dev_preprocessed.txt"

CUDA_VISIBLE_DEVICES=6 python main.py \
    --train_dir ${TRAIN_DATA}\
    --output_dir ${OUTPUT_DIR} \
    --max_length ${LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUM} \
    --num_train_epochs ${N_EPOCHS} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LR} \
    --save_total_limit ${STL} \
    --tag ${TAG} \
    --model t5-base \
    --sample_num 75000 \
    --seed ${RANDOM_SEED}  #1>>./result/${TAG}.txt 2>&1
    #--debug --toy
    #--dev_data_path ${DEV_DATA} \
