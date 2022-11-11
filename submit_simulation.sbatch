#!/bin/bash
#SBATCH -J ic_ssnet_sim_round2
#SBATCH -c 1 # Number of cores
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --mem 32000
#SBATCH -t 0-36:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_round2//output/ic_ssnet_sim__%A_%a.out # Standard output
#SBATCH -e /n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_round2//error/ic_ssnet_sim_%A_%a.err # Standard error
#SBATCH --array 1-100

N=10000
OFFSET=9999
export PYTHONPATH=/n/home10/felixyu/hebe/:$PYTHONPATH
OUTPUT_PREFIX=/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_round2/
source /n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/setup.sh
CMD="python /n/home10/felixyu/hebe/examples/fasrc_example_ppc.py --output_prefix ${OUTPUT_PREFIX}/data/ --lic_location ${OUTPUT_PREFIX}/lic/config.lic -n ${N} --padding 100 --emin 100 -s ${OFFSET}${SLURM_ARRAY_TASK_ID} --no_plot"
echo ${CMD}
$CMD