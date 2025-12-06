#!/bin/bash

#######################################################
# ProAssist Training Runner - Following standard pattern
#######################################################

set -e

# --- Environment Detection ---
if [ -n "$SLURM_JOB_ID" ]; then
    ENVIRONMENT="SLURM_EXECUTE"
    IS_SLURM=true
elif command -v sbatch &> /dev/null; then
    ENVIRONMENT="SLURM_SUBMIT"
    IS_SLURM=true
else
    ENVIRONMENT="LOCAL_RUN"
    IS_SLURM=false
fi

echo "🔍 Detected Environment: $ENVIRONMENT"

# --- Configuration ---
setting="$1"
interactive="i"

if [ "$IS_SLURM" = true ]; then
    # SLURM Configuration
    if [[ "$setting" == "$interactive" ]]; then
        partition='gpuA40x4-interactive'
        time='4:00:00'
        num_gpus=1
    else
        partition='gpuA40x4'
        time='1-00:00:00'
        num_gpus=1
    fi
    memory=64g

    # Delta paths
    PROJECT_ROOT=/scratch/bbyl/amosharrof/ProAssist
    CONDA_ENV_PATH=$PROJECT_ROOT/.venv

    # Setup SLURM output folders
    d_folder=$(date +'%Y-%m-%d')
    SLURM_FOLDER_BASE=slurm_out/proassist_training/
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

    # Change to project root
    cd "$PROJECT_ROOT"
    echo "📁 Current working directory: $(pwd)"

    # Set PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT/custom/src:${PYTHONPATH:-}"
    echo "📦 PYTHONPATH: $PYTHONPATH"
    
    # Set HF cache
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export TRANSFORMERS_CACHE="$HF_HOME"
}

# --- Execution ---
run_training() {
    echo "╔══════════════════════════════════════════════╗"
    echo "║   ProAssist Training${IS_SLURM:+ (SLURM)}   ║"
    echo "╚══════════════════════════════════════════════╝"
    echo "🐍 Using Python: $(which python)"
    echo "📦 PYTHONPATH: $PYTHONPATH"
    echo "📂 Running from: $(pwd)"
    echo ""

    # Run with accelerate for proper multi-GPU training (uses DistributedDataParallel)
    $CONDA_ENV_PATH/bin/accelerate launch --mixed_precision=bf16 custom/src/prospect/train/train_dst_proassist.py
}

# --- Main Execution Flow ---
if [ "$IS_SLURM" = true ] && [ "$ENVIRONMENT" = "SLURM_SUBMIT" ]; then
    # We're on login node, submit the job
    echo "📤 Submitting SLURM job..."
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=proassist_train
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --gpus=$num_gpus
#SBATCH --cpus-per-task=8
#SBATCH --mem=$memory
#SBATCH --time=$time
#SBATCH --output=$SLURM_FOLDER/train_%j.out
#SBATCH --error=$SLURM_FOLDER/train_%j.err

$(declare -f setup_environment)
$(declare -f run_training)

setup_environment
run_training
EOF

    echo "✅ Job submitted! Check output in: $SLURM_FOLDER"
    
else
    # We're either on compute node or local machine, run directly
    setup_environment
    run_training
    echo ""
    echo "✅ Training complete!"
fi
