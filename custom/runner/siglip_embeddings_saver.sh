#!/bin/bash

#######################################################
# SigLIP Embeddings Saver Runner - For Delta and Local
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
        time='1:00:00'
        num_gpus=1
    else
        partition='gpuA40x4'
        time='2-00:00:00'
        num_gpus=4
    fi
    memory=200g

    # Delta paths
    PROJECT_ROOT=/scratch/bbyl/amosharrof/ProAssist
    CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv

    # Logs
    d_folder=$(date +'%Y-%m-%d')
    SLURM_FOLDER_BASE=slurm_out/siglip_embeddings/
    mkdir -p $SLURM_FOLDER_BASE/$d_folder
    SLURM_FOLDER=$SLURM_FOLDER_BASE/$d_folder
else
    # Local Configuration
    PROJECT_ROOT=/u/siddique-d1/adib/ProAssist
    CONDA_ENV_PATH=$PROJECT_ROOT/.venv
fi

# --- The Fixed Environment Setup ---
setup_environment() {
    echo "🔧 Setting up environment..."

    # 1. ACTIVATE CONDA (The NCSA Delta Way)
    if [ -f "/sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh" ]; then
        echo "✓ Found NCSA Delta Conda Configuration"
        source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV_PATH"
        
        # Load helper modules
        module load libaio/0.3.113 2>/dev/null || true
        
    # Fallback for Local Machine
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_PATH"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_PATH"
    else
        # Last resort
        source activate "$CONDA_ENV_PATH" || conda activate "$CONDA_ENV_PATH"
    fi

    # 2. Project Setup
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/custom/src:${PYTHONPATH:-}"
    
    echo "✅ Active Python: $(which python)"
    echo "📦 PYTHONPATH: $PYTHONPATH"
}

# --- Execution Function ---
run_siglip_embeddings() {
    echo ""
    echo "🚀 Starting SigLIP Embeddings Extraction..."
    echo "📂 Running from: $(pwd)"
    echo "🐍 Python module: dst_data_builder.siglip_embeddings_saver"
    echo "📁 Project root: $PROJECT_ROOT"
    echo ""

    # Run the SigLIP embeddings saver with project_root override
    python -m dst_data_builder.siglip_embeddings_saver project_root=$PROJECT_ROOT

    echo ""
    echo "✅ SigLIP embeddings extraction completed successfully!"
}

# --- Main Logic ---
if [ "$ENVIRONMENT" == "SLURM_SUBMIT" ]; then
    echo "🚀 Submitting SLURM job to partition: $partition"
    sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time
#SBATCH --job-name=siglip_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -e $SLURM_FOLDER/%j.err
#SBATCH -o $SLURM_FOLDER/%j.out
#SBATCH -A bbyl-delta-gpu
#SBATCH --partition=$partition
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'

# Pass variables to the job
PROJECT_ROOT=$PROJECT_ROOT
CONDA_ENV_PATH=$CONDA_ENV_PATH

# Export functions to subshell
$(declare -f setup_environment)
$(declare -f run_siglip_embeddings)

setup_environment
run_siglip_embeddings
exit 0
EOT

else
    # Running Interactively or Locally
    setup_environment
    run_siglip_embeddings
fi
