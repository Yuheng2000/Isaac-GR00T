export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=GR00T-N1-Reproduction
export WANDB_API_KEY="327df07cca0b4918195078945539c819acc6ac97"
export WANDB_RUN_NAME=GR00T-N1-demo-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

CUDA_VISIBLE_DEVICES=7 python /home/jiyuheng/Isaac-GR00T/scripts/finetune_demo_g1.py