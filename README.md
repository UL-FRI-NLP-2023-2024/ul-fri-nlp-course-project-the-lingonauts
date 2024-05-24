# Natural language processing course 2023/24: LLM Prompt Strategies for Commonsense-Reasoning Tasks

Repository containing all three submissions for course Natural Language Processing.

## Repository general overview

### *report* 

Contains the final LaTeX/PDF report for the assignment.

### *report/code*

Contains two directories, containing code for both used datasets - CommonsenseQA and CommonGen.

### *report/code/commonsenseQA*

Contains code for each prompting strategy and parsing of code output.
Also contains directories for raw code output and parsed answers.

### *report/code/common_gen*

Contains code for each of the prompting strategies. 
Also contains directories for raw code output and answer evaluation.

## HPC first time setup

Clone the repository via SSH to the HPC home directory. (You can use VSCode or manually)

1. Go the directory
```shell
cd ul-fri-nlp-course-project-the-lingonauts/
```

2. Create a virtual environment
```shell
python3.11 -m venv .venv
```

3. If not activated yet, activate the virtual environment (note: VSCode will detect the virtual environment automatically)
```shell
source .venv/bin/activate
```

4. Change shell to a SLURM node (I think it defaults to 2 CPU cores, you can use -c # flag to allocate more)
```shell

salloc
```

5. Install requirements
```shell
srun pip install -r requirements.txt
```

6. Exit the SLURM node via Ctrl+D or
```shell
exit
```

7. Run the example code
```shell
salloc --partition=gpu --mem=64G -G 1 -c 16
srun python report/code/example.py
```

## Running batch jobs on HPC
1. Make sure you did steps 1. to 6. in the **HPC first time setup**.  

2. Go to report/code subdirectory
```shell
cd report/code/
```

3. Open _report/code/run_python.sh_ in an editor like nano/vim or just show it.    
```shell
cat run_python.sh
```

4. Review the SBATCH settings (job name and output file).

5. Make sure run_python.sh is executable
```shell
[mb62860@hpc-login2 code]$ ls -la run_python.sh 
-rwxr-xr-x 1 mb62860 mb62860 760 May  1 11:53 run_python.sh
```
```shell
# If you don't see the x bytes set, do
chmod +x run_python.sh
```
6. Run the script with the sbatch command. Provide the path to the python script right after run_python.sh
```shell
sbatch run_python.sh example.py
```
7. If you are doing multiple runs at the same time, modify the sbatch settings for each task, so they don't override the logs. e.g.:
```shell
sbatch --output=example.log --job-name=lingonauts_example run_python.sh example.py
sbatch --output=test.log --job-name=lingonauts_test run_python.sh test.py
sbatch --output=foo.log --job-name=lingonauts_foo run_python.sh foo.py
sbatch --output=bar.log --job-name=lingonauts_bar run_python.sh bar.py
```