#!/bin/bash

#######################################################
# DST Training Runner - Dynamic for Local/SLURM environments
# Runs the DST training pipeline with SmolVLM2
# Uses frame-level multi-task learning
#######################################################

set -e

# --- Environment Detection ---
# Check if we're on a SLURM system
if command -v sbatch &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    IS_SLURM=true
    echo "🔍 Detected SLURM environment"
else
    IS_SLURM=false
    echo "🔍 Detected local environment"
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DST Training Runner (SmolVLM2)${IS_SLURM:+ (SLURM)}   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# --- Configuration Setup ---
# Allow overriding project root via command line argument
if [ $# -gt 0 ]; then
    OVERRIDE_PROJECT_ROOT="$1"
    shift  # Remove the first argument so remaining args go to training
fi

if [ "$IS_SLURM" = true ]; then
    # SLURM Configuration - use override if provided, otherwise default
    PROJECT_ROOT="${OVERRIDE_PROJECT_ROOT:-/scratch/bbyl/amosharrof/ProAssist}"
    partition='gpuA40x4'
    time='2-00:00:00'
    memory=200g
    num_gpus=2

    # Delta cluster paths
    CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv

    # Setup SLURM output folders
    d_folder=$(date +'%Y-%m-%d')
    SLURM_FOLDER_BASE=slurm_out/training/
    mkdir -p $SLURM_FOLDER_BASE/$d_folder
    SLURM_FOLDER=$SLURM_FOLDER_BASE/$d_folder
else
    # Local machine configuration - use override if provided, otherwise explicit project root
    PROJECT_ROOT="${OVERRIDE_PROJECT_ROOT:-/u/siddique-d1/adib/ProAssist}"
    CONDA_ENV_PATH="${CONDA_ENV_PATH:-$PROJECT_ROOT/.venv}"  # Use CONDA_ENV_PATH env var or default

    echo -e "${GREEN}📁 Local project root: $PROJECT_ROOT${NC}"
    echo -e "${GREEN}🐍 Conda env path: $CONDA_ENV_PATH${NC}"
fi

# --- Environment Setup ---
setup_environment() {
    # Source environment
    if [ -f ~/.bash_profile ]; then
        source ~/.bash_profile
        echo -e "${GREEN}✓ Sourced: ~/.bash_profile${NC}"
    fi
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
        echo -e "${GREEN}✓ Sourced: ~/.bashrc${NC}"
    fi

    # Handle HOME change for conda (if needed)
    if [ -f ~/.bash_profile ]; then
        cd ~ && source ~/.bash_profile > /dev/null 2>&1
        echo -e "${GREEN}✓ Sourced (after HOME change): ~/.bash_profile${NC}"
    fi

    # Set HuggingFace cache to avoid disk space issues
    export HF_HOME="${HOME}/.cache/huggingface"
    mkdir -p "${HF_HOME}"
    echo -e "${GREEN}📦 HF_HOME: ${HF_HOME}${NC}"

    # GPU configuration
    if [ "$IS_SLURM" = true ]; then
        export CUDA_VISIBLE_DEVICES="0,1"
        echo -e "${GREEN}📦 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}${NC}"
    fi

    # Check API keys
    if [ -n "$OPENAI_API_KEY" ]; then
        KEY_DISPLAY="${OPENAI_API_KEY:0:6}...${OPENAI_API_KEY: -4}"
        echo -e "${GREEN}🔑 OPENAI_API_KEY found: $KEY_DISPLAY${NC}"
    fi

    # Change to project root
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}📁 Current working directory: $(pwd)${NC}"

    # Set PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT/custom/src:$PROJECT_ROOT:${PYTHONPATH:-}"
    echo -e "${GREEN}📦 PYTHONPATH: $PYTHONPATH${NC}"
}

# --- Training Execution ---
run_training() {
    echo ""
    echo -e "${BLUE}🚀 Starting DST Training (Multi-GPU with Accelerate)...${NC}"
    echo -e "${BLUE}📂 Running from: $(pwd)${NC}"
    echo ""

    # Build command with Accelerate launcher
    ACCELERATE_CMD="$CONDA_ENV_PATH/bin/accelerate"
    CMD="$ACCELERATE_CMD launch --mixed_precision=bf16 custom/src/prospect/train/dst_training_prospect.py"

    # Add any remaining CLI arguments passed to this script
    if [ $# -gt 0 ]; then
        CMD="$CMD $@"
    fi

    # Run the training command
    eval "$CMD"

    # Check exit code
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ DST Training completed successfully!${NC}"
        echo -e "${GREEN}📁 Check outputs in: custom/outputs/dst_training/${NC}"
        echo -e "${GREEN}📝 Stdout/stderr saved to: <hydra_output_dir>/training_stdout.log${NC}"
    else
        echo ""
        echo -e "${RED}❌ DST Training failed with exit code $EXIT_CODE${NC}"
        exit 1
    fi
}

# --- Main Logic ---
if [ "$IS_SLURM" = true ]; then
    # Submit SLURM job
    echo -e "${BLUE}🚀 Submitting SLURM job...${NC}"
    sbatch <<EOT
#!/bin/bash
#SBATCH --mem=$memory
#SBATCH --time=$time
#SBATCH --job-name=dst_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -e $SLURM_FOLDER/%j.err
#SBATCH -o $SLURM_FOLDER/%j.out
#SBATCH -A bbyl-delta-gpu
#SBATCH --partition=$partition
#SBATCH --gpus-per-node=$num_gpus
#SBATCH --constraint='scratch'

# Environment setup on compute node
$(declare -f setup_environment)

# Parse optional project root override (first argument)
if [ \$# -gt 0 ]; then
    PROJECT_ROOT="\$1"
    shift  # Remove the first argument so remaining args go to training
fi

setup_environment

$(declare -f run_training)
run_training "\$@"

exit 0
EOT
else
    # Run locally
    echo -e "${BLUE}🚀 Running locally...${NC}"

    # Parse optional project root override (first argument)
    if [ $# -gt 0 ]; then
        PROJECT_ROOT="$1"
        shift  # Remove the first argument so remaining args go to training
        echo -e "${YELLOW}🔧 Using project root override: $PROJECT_ROOT${NC}"
    fi

    setup_environment
    run_training "$@"
fi
