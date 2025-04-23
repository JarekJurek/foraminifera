#!/bin/bash
### --------------- job name ------------------
#BSUB -J forams

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=4GB]"

### --------------- wall-clock time ---------------
#BSUB -W 12:00

### --------------- output and error files ---------------
#BSUB -o forams_%J.out
#BSUB -e forams_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/DLCV/adlcv-ex-1/venv/bin/activate
export WANDB_API_KEY=90603e7b8caa45fca6a820844b7eb700a72aa61a
python3 -m src.train
