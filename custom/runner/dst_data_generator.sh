#!/bin/bash

# DST Data Generator Runner - Dynamic for Local/SLURM environments
# Automatically detects environment and adapts configuration

# --- Environment Detection ---

# Check 1: Are we running as a SLURM job (on a compute node)?
# This check is necessary because the script runs itself on the compute node 
# after submission and must distinguish the execution phase from the submission phase.
if [ -n "$SLURM_JOB_ID" ]; then
    ENVIRONMENT="SLURM_EXECUTE"
    IS_SLURM=true
    echo "🔍 Detected COMPUTE NODE (Job ID: $SLURM_JOB_ID)"

# Check 2: Can we submit a SLURM job? This is the simplified check for the Login Node.
# We consolidate the checks (sbatch exists is enough to confirm cluster context).
elif command -v sbatch &> /dev/null; then
    ENVIRONMENT="SLURM_SUBMIT"
    IS_SLURM=true
    echo "🔍 Detected CLUSTER LOGIN NODE (Ready for submission)"

else
    # Default case: Running on a local/unrelated development machine.
    ENVIRONMENT="LOCAL_RUN"
    IS_SLURM=false
    echo "🔍 Detected LOCAL DEVELOPMENT ENVIRONMENT"
fi

# --- Configuration Setup ---
setting="$1"
interactive="i"

if [ "$IS_SLURM" = true ]; then
    # SLURM Configuration
    if [[ "$setting" == "$interactive" ]]; then
        partition='gpuA40x4-interactive'
        time='1:00:00'
        num_gpus=1
    else
        partition='gpuA40x4'
        time='2-00:00:00'
        num_gpus=1
    fi
    memory=200g

    # Delta cluster paths
    PROJECT_ROOT=/scratch/bbyl/amosharrof/ProAssist
    CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv

    # Setup SLURM output folders
    d_folder=$(date +'%Y-%m-%d')
    SLURM_FOLDER_BASE=slurm_out/dst_gen/
    mkdir -p $SLURM_FOLDER_BASE/$d_folder
    SLURM_FOLDER=$SLURM_FOLDER_BASE/$d_folder
else
    # Local machine configuration
    PROJECT_ROOT=/u/siddique-d1/adib/ProAssist
    CONDA_ENV_PATH=$PROJECT_ROOT/.venv

fi

# --- Environment Setup ---
setup_environment() {
    # Source bash profile if it exists
    if [ -f ~/.bash_profile ]; then
        source ~/.bash_profile
        echo "Sourced: ~/.bash_profile"
    fi
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
        echo "Sourced: ~/.bashrc"
    fi

    # Handle HOME change for conda (if needed)
    if [ -f ~/.bash_profile ]; then
        cd ~ && source ~/.bash_profile > /dev/null 2>&1
        echo "Sourced (after HOME change): ~/.bash_profile"
    fi

    # Change to project root
    cd "$PROJECT_ROOT"
    echo "📁 Current working directory: $(pwd)"

    # Set PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT/custom/src:/mounts/u-amo-d1/adibm-data/projects/ZSToD/src:${PYTHONPATH:-}"
    echo "📦 PYTHONPATH: $PYTHONPATH"
}

# --- Execution ---
run_generator() {
    echo "╔══════════════════════════════════════════════╗"
    echo "║   DST Data Generator${IS_SLURM:+ (SLURM)}   ║"
    echo "╚══════════════════════════════════════════════╝"
    echo "🐍 Using Python: $(which python)"
    echo "📦 PYTHONPATH: $PYTHONPATH"
    echo "📂 Running from: $(pwd)"
    echo ""

    # Run the generator module
    python -m dst_data_builder.simple_dst_generator

    echo ""
    echo "✅ DST data generation completed successfully!"
}

# --- Main Logic ---
if [ "$IS_SLURM" = true ]; then
    # Submit SLURM job
    echo "🚀 Submitting SLURM job..."
    sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time
#SBATCH --job-name=dst_generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -e $SLURM_FOLDER/%j.err
#SBATCH -o $SLURM_FOLDER/%j.out
#SBATCH -A bbyl-delta-gpu
#SBATCH --partition=$partition
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'

# Environment setup on compute node
source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate $CONDA_ENV_PATH

$(declare -f setup_environment)
setup_environment

$(declare -f run_generator)
run_generator

exit 0
EOT
else
    # Run locally
    echo "🚀 Running locally..."
    setup_environment
    run_generator
fi