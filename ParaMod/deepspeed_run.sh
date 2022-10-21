LENGTH=512
BATCH_SIZE=4  # batch size per each GPU
LR="1e-4"
ACCUM=4
N_EPOCHS=30
SAVE_STEPS=10000  # Validating and save checkpoints
RANDOM_SEED=1234
STL=5
TAG="ds-test"

TRAIN_DATA='/home/ljx/Megatron-LM-main/tests/new_t5/output/new_bookcorpus/run_311'
OUTPUT_DIR='/data/ljx/result/para_model/{}/checkpoints/'
#DEV_DATA="./data/QQP_split/dev_preprocessed.txt"

deepspeed --include localhost:0,6,7 --master_port 61000 \
main.py \
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
    --model t5-large \
    --seed ${RANDOM_SEED} \
    --fp16 \
    --deepspeed deepspeed/stage3.json
    
    
    #1>>./result/${TAG}.txt 2>&1
    #--debug --toy
    #--dev_data_path ${DEV_DATA} \
