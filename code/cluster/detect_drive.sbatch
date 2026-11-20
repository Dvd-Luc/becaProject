#!/bin/bash
# Run detect_drive.py (full survey, step 11) on one drive's copy on the MeSU cluster.
#
#   sbatch --export=ALL,DRIVE=D4,EXTRA="--limit 64" cluster/detect_drive.sbatch   # short test job first
#   sbatch --export=ALL,DRIVE=D4 cluster/detect_drive.sbatch                      # the whole drive
#   myqueue; tail -f logs/beca_detect_<jobid>.out; suff <jobid>                  # follow it
#
# One job = one drive on one whole thin node (64 cores). The audio is read from
# $SCRATCH/beca/audio/<DRIVE>/ (an rsync copy of the drive, same folder layout); results
# go to drive_detection.dir as on the laptop. The run resumes: if the job hits the
# walltime, or the drive was not fully uploaded yet, submit the same line again.
#
# Needs: the repo cloned in $HOME/becaProject, generations/gen_2/model.tflite copied
# there, the venv $HOME/becavenv (pip install -r cluster/requirements.txt) and the
# logs/ folder. Setup steps: README, step 11, "Running on the MeSU cluster".
#
# Writes: logs/beca_detect_<jobid>.out (this job's output), plus detect_drive.py's outputs.
#
# Submit from code/.
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=23:00:00
#SBATCH -J beca_detect
#SBATCH -o logs/%x_%j.out

set -euo pipefail

PYTHON_MODULE=python/3.12          # exact name from `module avail python` on the cluster
VENV=$HOME/becavenv

if [ -z "${DRIVE:-}" ]; then
    echo "error: no drive given; submit with sbatch --export=ALL,DRIVE=D4 cluster/detect_drive.sbatch"
    exit 1
fi

module purge
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
cd "$HOME/becaProject/code"

echo "# $(date '+%Y-%m-%d %H:%M:%S') drive $DRIVE on $(hostname), $SLURM_CPUS_PER_TASK cores, job $SLURM_JOB_ID"
echo "# commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src config || echo ' (uncommitted changes)')"

python src/survey/detect_drive.py --drive "$DRIVE" --root "$SCRATCH/beca/audio/$DRIVE" \
       --threads "$SLURM_CPUS_PER_TASK" ${EXTRA:-}
