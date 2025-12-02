#!/bin/bash

#######################################################
# DST Training Runner - Dynamic for Local/SLURM environments
# Runs the DST training pipeline with SmolVLM2
# Uses frame-level multi-task learning
#######################################################

set -e

# --- Environment Detection ---

# Check 1: Are we running as a SLURM job (on a compute node)?
# This check is necessary because the script runs itself on the compute node 
# after submission and must distinguish the execution phase from the submission phase.
if [ -n "$SLURM_JOB_ID" ]; then
    ENVIRONMENT="SLURM_EXECUTE"
    IS_SLURM=true
    echo -e "🔍 ${GREEN}Detected COMPUTE NODE (Job ID: $SLURM_JOB_ID)${NC}"

# Check 2: Can we submit a SLURM job? This is the simplified check for the Login Node.
# We consolidate the checks (sbatch exists is enough to confirm cluster context).
elif command -v sbatch &> /dev/null; then
    ENVIRONMENT="SLURM_SUBMIT"
    IS_SLURM=true
    echo -e "🔍 ${BLUE}Detected CLUSTER LOGIN NODE (Ready for submission)${NC}"

else
    # Default case: Running on a local/unrelated development machine.
    ENVIRONMENT="LOCAL_RUN"
    IS_SLURM=false
    echo -e "🔍 ${YELLOW}Detected LOCAL DEVELOPMENT ENVIRONMENT${NC}"
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DST Training Runner (SmolVLM2)${IS_SLURM:+ (SLURM)}    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# --- Configuration Setup ---
# Parse arguments: first argument can be project root override or "i" for interactive
OVERRIDE_PROJECT_ROOT=""
setting=""

if [ $# -gt 0 ]; then
    if [[ "$1" == "i" ]]; then
        setting="$1"
    else
        OVERRIDE_PROJECT_ROOT="$1"
    fi
    shift
fi

interactive="i"

if [ "$ENVIRONMENT" != "LOCAL_RUN" ]; then
    # SLURM Configuration (applies to both SUBMIT and EXECUTE phases)
    if [[ "$setting" == "$interactive" ]]; then
        partition='gpuA100x4-interactive'
        time='1:00:00'
        num_gpus=1
    else
        partition='gpuA100x4'
        time='2-00:00:00'
        num_gpus=4
    fi
    memory=200g

    # Assuming scratch is the high-performance file system on the cluster
    PROJECT_ROOT="${OVERRIDE_PROJECT_ROOT:-/scratch/bbyl/amosharrof/ProAssist}"
    CONDA_ENV_PATH=/scratch/bbyl/amosharrof/ProAssist/.venv

    # Setup SLURM output folders (This is done once, during submission)
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
    # The 'cd ~' here is generally unnecessary and can be removed, 
    # but kept for safety if you need it to load profile variables correctly.
    if [ -f ~/.bash_profile ]; then
        cd ~ && source ~/.bash_profile > /dev/null 2>&1
        echo -e "${GREEN}✓ Sourced (after HOME change): ~/.bash_profile${NC}"
    fi

    # Set HuggingFace cache to avoid disk space issues
    # Use $HOME if running on compute node, otherwise rely on the default if local/dev
    export HF_HOME="${HOME:-$PROJECT_ROOT}/.cache/huggingface"
    mkdir -p "${HF_HOME}"
    echo -e "${GREEN}📦 HF_HOME: ${HF_HOME}${NC}"

    # GPU configuration
    if [ "$ENVIRONMENT" == "SLURM_EXECUTE" ]; then
        # On compute nodes, SLURM sets $SLURM_GPUS_ON_NODE.
        # We manually set CUDA_VISIBLE_DEVICES for PyTorch/Accelerate to the right GPUs
        # NOTE: Using all allocated GPUs (0 through $num_gpus-1)
        if [ "$num_gpus" -gt 0 ]; then
            GPU_INDICES=$(seq -s, 0 $((num_gpus - 1)))
            export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
            echo -e "${GREEN}📦 CUDA_VISIBLE_DEVICES set to $GPU_INDICES (from sbatch config)${NC}"
        fi
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

if [ "$ENVIRONMENT" == "SLURM_SUBMIT" ]; then
    # Submit SLURM job (Runs on Login Node)
    echo -e "${BLUE}🚀 Submitting SLURM job...${NC}"
    
    # Preserve original arguments passed to the script for the job script
    ORIGINAL_ARGS="$@"

    sbatch <<'EOT'
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

# Load environment setup functions onto the compute node
$(declare -f setup_environment)
setup_environment

# Load training execution function onto the compute node
$(declare -f run_training)
# Execute the training function with the original arguments
run_training "$ORIGINAL_ARGS"

exit 0
EOT

elif [ "$ENVIRONMENT" == "SLURM_EXECUTE" ]; then
    # Run on Compute Node (This is the execution path inside the submitted job)
    echo -e "${GREEN}Starting Compute Node tasks...${NC}"
    
    # Environment setup is called again inside the job script template
    # by the Sbatch wrapper, so it's ready.
    
    # Run training with arguments passed from the sbatch wrapper
    run_training "$@"

else # LOCAL_RUN
    # Run locally (for quick debugging/setup)
    echo -e "${YELLOW}Running locally...${NC}"

    if [ -n "$OVERRIDE_PROJECT_ROOT" ]; then
        echo -e "${YELLOW}🔧 Using project root override: $OVERRIDE_PROJECT_ROOT${NC}"
    fi

    setup_environment
    run_training "$@"
fi

echo "Script finished on $ENVIRONMENT."