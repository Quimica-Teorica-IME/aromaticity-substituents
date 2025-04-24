import os

root_dir = os.getcwd()

# Conteúdo do arquivo .input e .sub
template_submission = """#!/bin/bash
#SBATCH --job-name={molecule}_top-1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=720:00:00

module load gdma/2.3.3

export SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
gdma < gdma_{molecule}.input > gdma_{molecule}.output
"""
template_input = """File {molecule}.fchk
Angstrom
Multipoles
  Limit 2
  Switch 4.0
  Radius H 0.35
  Punch file.punch
  ADD TOP(-1) 0.0 0.0 -1.00
Start
Finish
"""

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):

        molecule_name = folder

        input_file = os.path.join(folder_path, f"gdma_{molecule_name}.input")
        sub_file = os.path.join(folder_path, f"gdma.sub")

        # Escrever o arquivo .input
        with open(input_file, 'w', newline='\n') as file:
            file.write(template_input.format(molecule=molecule_name))

        # Escrever o arquivo .sub
        with open(sub_file, 'w', newline='\n') as file:
            file.write(template_submission.format(molecule=molecule_name))
