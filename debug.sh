# conda activate /home/jiyuheng/env/gr00t/
source /home/anpengju/miniconda3/bin/activate /home/anpengju/envs/gr00t

BASE_MODEL=/share/project/jiyuheng/groot_n1
DATASET=/share/project/jiyuheng/Isaac-GR00T/demo_data/G1_Dex3_Pick_up_the_bread_that_is_closer_to_the_cup_and_put_it_into_the_plate
EXP_NAME=train_g1_task2_ve
OUTPUT_DIR=/share/project/apj/runs/N1/${EXP_NAME}
ROBOT_CONFIG=baai_g1_dex3

export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=offline

TUNE_LLM="--no-tune-llm"
TUNE_VISUAL="--tune-visual"
TUNE_PROJECTOR="--tune-projector"
TUNE_DIFFUSION_MODEL="--tune-diffusion-model"

MAX_STEP=50000
NUM_GPU=4

BATCHSIZE=8
SAVE_STEP=10000
LR=0.0001
WEIGHT_DECAY=1e-05
WARMUP=0.05

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

# CUDA_VISIBLE_DEVICES=7

cd /home/anpengju/Isaac-GR00T
export PYTHONPATH=`pwd`:$PYTHONPATH

python scripts/gr00t_finetune.py \
    --data_config $ROBOT_CONFIG \
    --dataset-path $DATASET \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCHSIZE \
    --max-steps $MAX_STEP \
    --save-steps $SAVE_STEP \
    --num-gpus $NUM_GPU \
    --base-model-path $BASE_MODEL \
    $TUNE_LLM \
    $TUNE_VISUAL \
    $TUNE_PROJECTOR \
    $TUNE_DIFFUSION_MODEL \
    --no-resume \
    --learning-rate $LR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-ratio $WARMUP \
    --dataloader-num-workers 1 \
    --embodiment-tag new_embodiment