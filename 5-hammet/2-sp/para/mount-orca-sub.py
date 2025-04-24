import os

root_dir = os.getcwd()

# Conteúdo do arquivo .input e .sub
template_submission = """#!/bin/bash
#SBATCH --job-name={molecule}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=720:00:00


module load orca/6.0.1.avx2

# Definir variáveis importantes
export ORCA_INPUT="{molecule}.inp"          # Nome do arquivo de entrada
export ORCA_OUTPUT="{molecule}.out"         # Nome do arquivo de saída
export ORCA_SCRDIR=$SLURM_JOB_ID             # Diretório temporário para arquivos de scratch

# Criar diretório de scratch
mkdir -p $ORCA_SCRDIR

# Executar o ORCA
orca $ORCA_INPUT > $ORCA_OUTPUT

# Mover arquivos de scratch de volta para o diretório de trabalho
mv $ORCA_SCRDIR/* .

# Limpar diretório de scratch
rm -rf $ORCA_SCRDIR

"""

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):

        molecule_name = folder

        sub_file = os.path.join(folder_path, f"orca.slurm")

        # Escrever o arquivo .sub
        with open(sub_file, 'w', newline='\n') as file:
            file.write(template_submission.format(molecule=molecule_name))
