# *[Specify the config file path and the GPU devices to use]
# export CUDA_VISIBLE_DEVICES=0

# *[Specify the config file path]
export REFANY3D_CONFIG=./train/config/train_refany3d.yaml


# *[Specify the WANDB API key]
# export WANDB_API_KEY=""
export WANDB_NAME="refany3d_train_$(date +%m%d%H%M)"

export LOG_FILE="logs/$WANDB_NAME.log"

mkdir -p logs

accelerate launch --main_process_port 12345 --num_processes 8 -m model.train_flux.train_refany3d 2>&1 | tee $LOG_FILE
