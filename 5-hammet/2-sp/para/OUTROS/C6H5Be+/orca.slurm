#!/bin/bash
#SBATCH --job-name=C6H5Be+
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=720:00:00


module load orca/6.0.1.avx2

# Definir vari�veis importantes
export ORCA_INPUT="C6H5Be+.inp"          # Nome do arquivo de entrada
export ORCA_OUTPUT="C6H5Be+.out"         # Nome do arquivo de sa�da
export ORCA_SCRDIR=$SLURM_JOB_ID             # Diret�rio tempor�rio para arquivos de scratch

# Criar diret�rio de scratch
mkdir -p $ORCA_SCRDIR

# Executar o ORCA
orca $ORCA_INPUT > $ORCA_OUTPUT

# Mover arquivos de scratch de volta para o diret�rio de trabalho
mv $ORCA_SCRDIR/* .

# Limpar diret�rio de scratch
rm -rf $ORCA_SCRDIR

