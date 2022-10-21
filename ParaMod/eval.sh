GPU_ID=4
TAG='wiki-t5base-base'
model_folder='t5_base'
#model_folder='t5_train_35_epochs_4-11'
#model_folder='t5-base-stage_2022-04-27'
#model_folder='t5-base_ft-2epoch-1000step_continued_2022-05-06'
#CHECKPOINT_DIR="/data/ljx/result/para_model/${model_folder}/checkpoints/checkpoint-31250" #$3
CHECKPOINT_DIR="/data/MODELS/t5-base"

T="1.0"  # temperature
k=10
p="1.0"
N_GEN=10
N_BEAM=15
SEED=1234

#dataset_folder='QQP_split'
#dataset_folder='mscoco_split'
#dataset_folder='twitter_split'
dataset_folder='wikianswers_split'

INPUT_FILE="./data/${dataset_folder}/test_input.txt"
PREPROCESSED="./data/${dataset_folder}/test_input.txt"
TARGET="./data/${dataset_folder}/test_target.txt"
FILENAME="inferenced_${TAG}_top-${k}-p${p}-T${T}_seed${SEED}.txt"

CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --data_path ${PREPROCESSED} \
    --checkpoint ${CHECKPOINT_DIR} \
    --save "/data/ljx/result/para_model/${model_folder}/inference/${FILENAME}" \
    --decoding "beam_gen" \
    --beam_size ${N_BEAM} \
    --model "t5-base" \
    --k ${k} \
    --p ${p} \
    --temperature ${T} \
    --num_generate ${N_GEN} \
    --seed ${SEED} \
    --tag ${TAG}

CUDA_VISIBLE_DEVICES=$GPU_ID python postprocessing.py \
    --input ${INPUT_FILE} \
    --paraphrase "/data/ljx/result/para_model/${model_folder}/inference/${FILENAME}" \
    --output "/data/ljx/result/para_model/${model_folder}/inference/new_filtered/${FILENAME}" \
    --model "bert-base-nli-stsb-mean-tokens" \
    --tag ${TAG}

CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
    --generated "/data/ljx/result/para_model/${model_folder}/inference/new_filtered/${FILENAME}" \
    --source ${INPUT_FILE} \
    --ground_truth ${TARGET} \
    --metrics 'bleu,ibleu,rouge,self-bleu,meteor' \
    --tag ${TAG}