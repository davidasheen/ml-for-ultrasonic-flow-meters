#!/bin/bash
#
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --output=vmpf0.%a_out
#SBATCH --error=vmpf0.%a_err


hcathresh=0.$SLURM_ARRAY_TASK_ID

while getopts f:n:t: flag; do
    case $flag in
        f) filename=$OPTARG;;
        n) n_jobs=$OPTARG;;
        #t) hcathresh=$OPTARG;;
    esac
done

echo "$filename";
echo "$n_jobs";
echo "$hcathresh";

source ~/venvs/pychemauth/bin/activate

echo "python vmpf_runner.py -f $filename -n $n_jobs -t $hcathresh"

python vmpf_runner.py -f $filename -n $n_jobs -t $hcathresh

exit 0
