#!/bin/bash
### --------------- job name ------------------
#BSUB -J 3d_train

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=8GB]"

### --------------- wall-clock time ---------------
#BSUB -W 24:00

### --------------- output and error files ---------------
#BSUB -o 3d_train_%J.out
#BSUB -e 3d_train_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/DLCV/adlcv-ex-1/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=90603e7b8caa45fca6a820844b7eb700a72aa61a
python3 -m src.train_prop