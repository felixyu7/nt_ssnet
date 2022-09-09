#!/bin/bash
#SBATCH -J ic_ssnet_sim
#SBATCH -c 1 # Number of cores
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem 6000
#SBATCH -t 0-10:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim//output/ic_ssnet_sim__%A_%a.out # Standard output
#SBATCH -e /n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim//error/ic_ssnet_sim_%A_%a.err # Standard error
#SBATCH --array 1-100

N=5000
export PYTHONPATH=/n/home10/felixyu/hebe/:$PYTHONPATH
OUTPUT_PREFIX=/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim/
source /n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/setup.sh
CMD="python /n/home10/felixyu/ic_ssnet/simulate.py --output_prefix ${OUTPUT_PREFIX}/data/ -n ${N} --lic_location ${OUTPUT_PREFIX}/lic/config.lic -s ${SLURM_ARRAY_TASK_ID}"
echo ${CMD}
$CMD
