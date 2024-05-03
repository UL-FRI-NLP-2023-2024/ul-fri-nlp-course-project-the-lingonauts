#!/bin/bash
#SBATCH --job-name=the-lingonauts-job     # the title of the job
#SBATCH --output=run_python.log           # file to which logs are saved
#SBATCH --time=24:00:00                   # job time limit - full format is D-H:M:S
#SBATCH --ntasks=1                        # number of tasks
#SBATCH --mem=64G                         # memory allocation
#SBATCH --partition=gpu                   # partition to run on nodes that contain gpus
#SBATCH --gpus=1                          # number of requested GPUs
#SBATCH --cpus-per-task=16                # number of allocated cores
set +x

# enter virtual environment (change directory if needed)
source ~/ul-fri-nlp-course-project-the-lingonauts/.venv/bin/activate

# run the python script provided
srun python $1
