#!/bin/bash -l

# --- SLURM Configuration Setup ---
setting="$1"
interactive="i"

# Determine SLURM partition and time based on interactive flag
if [[ "$setting" == "$interactive" ]]; then
    partition='gpuA40x4-interactive'
    time='1:00:00'
    num_gpus=1
else
    # Default to non-interactive A100 job (2 days limit)
    partition='gpuA40x4'
    time='2-00:00:00'
    num_gpus=1
fi
memory=200g # Use consistent memory allocation

# --- Project & Logging Setup ---
# Set the project root path for the remote machine (Delta's /scratch storage)
PROJECT_ROOT=/scratch/bbyl/amosharrof/ProAssist
CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv  # Path to your Conda env on Delta

# Setup SLURM output folders
d_folder=$(date +'%Y-%m-%d')
SLURM_FOLDER_BASE=slurm_out/dst_gen/
mkdir -p $SLURM_FOLDER_BASE/$d_folder
SLURM_FOLDER=$SLURM_FOLDER_BASE/$d_folder



# --- SBATCH Submission Block ---
sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time             # Time limit for the job (REQUIRED).
#SBATCH --job-name=dst_generate  # Job name based on the task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # Number of cores for the job
#SBATCH -e $SLURM_FOLDER/%j.err  # Error file for this job.
#SBATCH -o $SLURM_FOLDER/%j.out  # Output file for this job.
#SBATCH -A bbyl-delta-gpu        # Project allocation account name (REQUIRED)
#SBATCH --partition=$partition   # Partition/queue to run the job in. (REQUIRED)
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'   # Ensure access to /scratch storage

# --- Environment Setup on Compute Node ---

# 1. Source Conda environment
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate $CONDA_ENV_PATH

# 2. Change to project root on scratch storage
cd "$PROJECT_ROOT"

# 3. Set PYTHONPATH for the project code
# NOTE: The custom/src path is now relative to the remote PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/custom/src:${PYTHONPATH:-}"

echo "╔══════════════════════════════════════════════╗"
echo "║   DST Data Generator on Compute Node     ║"
echo "╚══════════════════════════════════════════════╝"
echo "🐍 Using Python: \$(which python)"
echo "📦 PYTHONPATH: \$PYTHONPATH"
echo "📂 Running from: \$(pwd)"
echo ""

# --- Execution ---

# Run the generator module
# Note: The local script used a variable for the python path,
# but using the activated 'python' and '-m' is cleaner on SLURM.
python -m dst_data_builder.simple_dst_generator 

echo ""
echo "✅ DST data generation completed successfully on SLURM!"

exit 0
EOT