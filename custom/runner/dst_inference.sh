#!/bin/bash

# DST Inference Runner - Dynamic for Local/SLURM environments
# Automatically detects environment and adapts configuration

# --- Environment Detection ---

# Check 1: Are we running as a SLURM job (on a compute node)?
if [ -n "$SLURM_JOB_ID" ]; then
    ENVIRONMENT="SLURM_EXECUTE"
    IS_SLURM=true
    echo "🔍 Detected COMPUTE NODE (Job ID: $SLURM_JOB_ID)"

# Check 2: Can we submit a SLURM job?
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
        time='2:00:00'
        num_gpus=1
    fi
    memory=200g

    # Delta cluster paths
    PROJECT_ROOT=/scratch/bbyl/amosharrof/ProAssist
    CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv

    # Setup SLURM output folders
    d_folder=$(date +'%Y-%m-%d')
    SLURM_FOLDER_BASE=slurm_out/dst_inference/
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
run_inference() {
    echo "╔══════════════════════════════════════════════╗"
    echo "║   DST Inference Runner${IS_SLURM:+ (SLURM)}      ║"
    echo "╚══════════════════════════════════════════════╝"
    echo "🐍 Using Python: $(which python)"
    echo "📦 PYTHONPATH: $PYTHONPATH"
    echo "📂 Running from: $(pwd)"
    echo ""

    export CUDA_VISIBLE_DEVICES="0"
    # Detect Available GPUs (Works for both Local and SLURM)
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Count commas + 1 to get number of devices
        num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        echo "   - CUDA_VISIBLE_DEVICES set. Using $num_gpus GPU(s)."
    elif command -v nvidia-smi &> /dev/null; then
        detected_gpus=$(nvidia-smi -L | wc -l)
        num_gpus=$detected_gpus
    elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
        num_gpus=$SLURM_GPUS_ON_NODE
    else
        num_gpus=${num_gpus:-1}
    fi
    
    echo "⚡ Launching inference on $num_gpus GPU(s) with bfloat16 precision..."

    # Configure Accelerate Launch Arguments
    if [ "$num_gpus" -gt 1 ]; then
        LAUNCH_ARGS="--multi_gpu --num_processes=$num_gpus"
        echo "   - Enabled: Multi-GPU DistributedDataParallel"
    else
        LAUNCH_ARGS="--num_processes=1"
        echo "   - Enabled: Single Process Mode"
    fi

    # Run the inference module with accelerate and bfloat16
    $CONDA_ENV_PATH/bin/accelerate launch \
        $LAUNCH_ARGS \
        --mixed_precision=bf16 \
        custom/src/prospect/inference/run_inference.py

    echo ""
    echo "✅ DST inference completed successfully!"
}

# --- Main Logic ---
if [ "$IS_SLURM" = true ]; then
    # Submit SLURM job
    echo "🚀 Submitting SLURM job..."
    sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time
#SBATCH --job-name=dst_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -e $SLURM_FOLDER/%j.err
#SBATCH -o $SLURM_FOLDER/%j.out
#SBATCH -A bbyl-delta-gpu
#SBATCH --partition=$partition
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'

# Set project root and environment variables for the job
PROJECT_ROOT=/scratch/bbyl/amosharrof/ProAssist
CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv

# Activate the virtual environment
if [ -f "$CONDA_ENV_PATH/bin/activate" ]; then
    source "$CONDA_ENV_PATH/bin/activate"
    echo "Activated virtual environment: $CONDA_ENV_PATH"
else
    echo "Warning: Could not find virtual environment at $CONDA_ENV_PATH"
fi

$(declare -f setup_environment)
setup_environment

$(declare -f run_inference)
run_inference

exit 0
EOT
else
    # Run locally
    echo "🚀 Running locally..."
    setup_environment
    run_inference
fi
