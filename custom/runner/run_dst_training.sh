#!/bin/bash

#######################################################
# DST Training Runner
# Runs the DST training pipeline with SmolVLM2
# Uses frame-level multi-task learning
#######################################################

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DST Training Runner (SmolVLM2)           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

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

# Use only GPU 0 for now to test single GPU training
export CUDA_VISIBLE_DEVICES="0,1"
echo -e "${GREEN}📦 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}${NC}"

# Check API keys
if [ -n "$OPENAI_API_KEY" ]; then
    KEY_DISPLAY="${OPENAI_API_KEY:0:6}...${OPENAI_API_KEY: -4}"
    echo -e "${GREEN}🔑 OPENAI_API_KEY found: $KEY_DISPLAY${NC}"
fi

# Get project root - run from /u/siddique-d1/adib/ProAssist
PROJECT_ROOT="/u/siddique-d1/adib/ProAssist"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi
echo -e "${GREEN}📁 Project root directory: $PROJECT_ROOT${NC}"

# Change to project root
cd "$PROJECT_ROOT"
echo -e "${GREEN}📁 Current working directory: $(pwd)${NC}"

# Look for venv
PYTHON_CMD="python3"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${GREEN}🔧 Found virtual environment at $PROJECT_ROOT/.venv${NC}"
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
    if [ -f "$VENV_PYTHON" ]; then
        PYTHON_CMD="$VENV_PYTHON"
        echo -e "${GREEN}✓ Using Python from venv: $VENV_PYTHON${NC}"
    else
        echo -e "${YELLOW}⚠️  Python not found in venv at $VENV_PYTHON, falling back to system python3${NC}"
    fi
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/custom/src:$PROJECT_ROOT:${PYTHONPATH:-}"
echo -e "${GREEN}📦 PYTHONPATH: $PYTHONPATH${NC}"
echo ""

echo -e "${BLUE}🚀 Starting DST Training (Multi-GPU with Accelerate)...${NC}"
echo -e "${BLUE}📂 Running from: $(pwd)${NC}"
echo ""

# Build command with Accelerate launcher
# Use full path to accelerate from venv (accelerate launch handles the python executable)
ACCELERATE_CMD="$PROJECT_ROOT/.venv/bin/accelerate"
CMD="$ACCELERATE_CMD launch --mixed_precision=bf16 custom/src/prospect/train/dst_training_prospect.py"

# Add any CLI arguments passed to this script
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
