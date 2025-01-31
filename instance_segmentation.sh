#! /bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH --job-name=hovernext_inference


source ~/.bashrc
mamba activate hovernext
python3 hovernext_inference.py -i /mnt/lustre-grete/usr/u12649/data/original_data -o /mnt/lustre-grete/usr/u12649/models/hovernext -d pannuke -c pannuke_convnextv2_tiny_3
