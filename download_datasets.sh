#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --mincpus=2
#SBATCH --mem=10000
#SBATCH --job-name=lc11downloader
#SBATCH --output=logs/downloader.txt
#SBATCH --error=logs/downloader.txt

source ~/.bash_profile
mkdir -p ~/.config/ && mkdir -p ~/.config/openml/ && echo 'cachedir=/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdb11/openml_cache/' > ~/.config/openml/config
conda activate lcdb11
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdb11/lcdb_function
python downloaddata.py
