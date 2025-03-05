#!/bin/bash
#SBATCH --job-name=qwen_train_on_latin_script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=node006
#SBATCH --gres=gpu:8
#SBATCH --time=168:00:00
#SBATCH --mem=700000M
#SBATCH --partition=defq
#SBATCH --output=slurm-%N.%j.out
#SBATCH --error=slurm-%N.%j.err

# Conda initialization
eval "$(conda shell.bash hook)"
conda activate small

# Environment variables for debugging and performance (still experimenting)
# export NCCL_IB_HCA=mlx5_0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=0
# export NCCL_IB_TIMEOUT=22
# export NCCL_DEBUG=WARN
# export NCCL_IB_RETRY_CNT=7
# export NCCL_IB_SL=0
# export OMP_NUM_THREADS=1
# export NCCL_IB_QPS_PER_CONNECTION=8

# Util environment variables
export SSL_CERT_FILE=/home/adal_abilbekov/cacert.pem
export WANDB_API_KEY=d6f39cbe461c04b7594d30e17320008239f80544
export WANDB_DISABLED=False
export HF_HOME=/scratch/adal_abilbekov
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1 
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Setting master address and port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

# Setting paths
TRAIN_SCRIPT="/home/adal_abilbekov/small_llm/nanotron/run_train.py"
CONFIG_PATH="/home/adal_abilbekov/small_llm/nanotron/configs/llama_500M_config.yaml"

# Debug information
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Current working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# distributed training command
srun torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_id $SLURM_JOB_ID \
    ${TRAIN_SCRIPT} \
    --config ${CONFIG_PATH}

# On error print
if [ $? -ne 0 ]; then
    echo "torchrun failed with exit code $?"
    exit 1
fi