# Natural language processing course 2023/24: LLM Prompt Strategies for Commonsense-Reasoning Tasks

Repository containing all three submissions for course Natural Language Processing.

## HPC setup

Clone the repository via SSH to the HPC home directory. (You can use VSCode or manually)

1. Go the directory
    $ cd ul-fri-nlp-course-project-the-lingonauts/
2. Create a virtual environment
    $ python3.11 -m venv .venv
3. If not activated yet, activate the virtual environment (note: VSCode will detect the virtual environment automatically)
    $ source .venv/bin/activate
4. Change shell to a SLURM node (I think it defaults to 2 CPU cores, you can use -c # flag to allocate more)
    $ salloc
5. Install requirements
    $ srun pip install -r requirements.txt
6. Exit the SLURM node via Ctrl+D or
    $ exit
7. Run the example code
    $ salloc --partition=gpu --mem=64G -G 1 -c 16
    $ srun python report/code/example.py