#!/bin/bash
#SBATCH --job-name=open_clip
#SBATCH --account=project_465000330
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --mem=448GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --open-mode=append
#SBATCH --partition=standard-g
#SBATCH --signal=USR2@120
#SBATCH --time=60
#SBATCH --wckey=submitit
#SBATCH --output=/scratch/project_465000727/repos/open_clip_fork/%j_0_log.out
#SBATCH --error=/scratch/project_465000727/repos/open_clip_fork/%j_0_log.err


module load LUMI/22.08
module load partition/G

echo "LD_LIBRARY_PATH"
echo $LD_LIBRARY_PATH
echo "PATH"
echo $PATH

export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3


  if [ $SLURM_LOCALID -eq 0 ] ; then
    ls /dev/shm/ | awk '!/data/' | xargs -i rm -rf  /dev/shm/{}
    rocm-smi || true
  else
    sleep 2
  fi

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12802


export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --unbuffered --output /scratch/project_465000727/repos/open_clip_fork/%j_%t_log.out --error /scratch/project_465000727/repos/open_clip_fork/%j_%t_log.err \
python -u src/training/main.py \
    --save-frequency 1 \
    --report-to wandb \
    --train-data="/scratch/project_465000727/datasets/img2dataset/mscoco/{00000..00058}.tar" \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=32 \
    --workers=7 \
    --model ViT-B-32 \
    --name "ViT-B-32-Vanilla" \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --train-num-samples 100000