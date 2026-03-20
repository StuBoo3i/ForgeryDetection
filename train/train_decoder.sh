#!/bin/bash
#SBATCH --job-name=train-decoder
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./outputs/decoder_%x_%j.out
#SBATCH --error=./outputs/decoder_%x_%j.err

echo "========== 作业信息 =========="
echo "提交目录: ${SLURM_SUBMIT_DIR}"
echo "工作目录: $PWD"
echo "运行节点: ${SLURM_NODELIST}"
echo "作业ID: ${SLURM_JOB_ID}"
echo "作业名称: ${SLURM_JOB_NAME}"
echo "================================"

eval "$(conda shell.bash hook)"

conda activate forgeryD

echo "Python 版本:"
python --version
echo "CUDA 可用性:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

PROJECT_DIR="/nfsdat/home/jwangslm/ForgeryDetection/train"
# 训练脚本路径
TRAIN_SCRIPT="${PROJECT_DIR}/train_decoder.py"
# 数据集根目录
DATA_ROOT="/nfsdat/home/bglvslm/competition/your_recon_dataset"
# 输出根目录
OUTPUT_DIR="${PROJECT_DIR}/outputs/decoder_train_${SLURM_JOB_ID}" 

# 自动创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/weights
mkdir -p ${OUTPUT_DIR}/recon_images

echo "开始训练，输出目录: ${OUTPUT_DIR}"
cd ${PROJECT_DIR} 

python ${TRAIN_SCRIPT} \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --device cuda \
    --early_stop_patience 10

if [ $? -eq 0 ]; then
    echo "训练成功完成！输出目录: ${OUTPUT_DIR}"
else
    echo "训练出错，请查看错误日志: ${OUTPUT_DIR}/../decoder_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi