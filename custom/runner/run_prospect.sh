#!/bin/bash

# Shell script to run PROSPECT Evaluator
# Usage: ./run_prospect.sh [hydra overrides...]
# Example: ./run_prospect.sh data_source.video_ids=[9011-c03f,P01_11] exp_name=multi_video

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PROSPECT Evaluator Runner           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Source bash profile to set correct HOME directory
if [ -f ~/.bash_profile ]; then
    source ~/.bash_profile
    echo -e "${GREEN}✅ Sourced ~/.bash_profile (HOME: $HOME)${NC}"
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Project root is two levels up from custom/runner
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Project root directory: ${PROJECT_ROOT}${NC}"
echo -e "${GREEN}📁 Current working directory: $(pwd)${NC}"

# Activate conda environment
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}🔧 Found conda environment at ${VENV_PATH}${NC}"
    
    # Try to activate; some setups prefer `conda activate`, others prefer conda run
    if command -v conda >/dev/null 2>&1; then
        # Initialize conda in this shell
        eval "$(conda shell.bash hook)" || true
        
        # Try activate; if it fails, we'll fall back to using `conda run` when invoking python
        if conda activate "$VENV_PATH" 2>/dev/null; then
            echo -e "${GREEN}🔁 Activated conda environment via 'conda activate'${NC}"
            ACTIVATED_VIA_ACTIVATE=1
        else
            echo -e "${YELLOW}⚠️  Could not 'conda activate' path; will use 'conda run -p' when executing python${NC}"
            ACTIVATED_VIA_ACTIVATE=0
        fi
    else
        echo -e "${YELLOW}⚠️  'conda' command not found in PATH. Will attempt to run python from the venv directly if available.${NC}"
        ACTIVATED_VIA_ACTIVATE=0
    fi
else
    echo -e "${RED}❌ Error: Conda environment not found at ${VENV_PATH}${NC}"
    echo -e "${YELLOW}Please create it (e.g. conda create -p ${VENV_PATH} python=3.10) or update the path in the script${NC}"
    exit 1
fi

# Add custom/src to PYTHONPATH for module imports
export PYTHONPATH="$PROJECT_ROOT/custom/src:$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}📦 PYTHONPATH: ${PYTHONPATH}${NC}"

# Build the Python command - run as module
PYTHON_CMD_ARGS="-m prospect.prospect_evaluator"

# Add any CLI arguments passed to this script
if [ $# -gt 0 ]; then
    PYTHON_CMD_ARGS="$PYTHON_CMD_ARGS $@"
fi

# Helper to run python either via activated conda or conda run -p or direct venv python
run_python() {
    if [ "${ACTIVATED_VIA_ACTIVATE:-0}" -eq 1 ]; then
        python $PYTHON_CMD_ARGS
    elif command -v conda >/dev/null 2>&1; then
        conda run -p "$VENV_PATH" --no-capture-output python $PYTHON_CMD_ARGS
    else
        # Try to use python binary inside the venv
        if [ -x "$VENV_PATH/bin/python" ]; then
            "$VENV_PATH/bin/python" $PYTHON_CMD_ARGS
        else
            echo -e "${RED}❌ No python executable found to run the evaluator${NC}"
            exit 1
        fi
    fi
}

# Show configuration info
echo ""
echo -e "${BLUE}Configuration: (from Hydra configs + CLI overrides)${NC}"
echo -e "  Base config: ${YELLOW}custom/config/prospect/prospect.yaml${NC}"
echo -e "  Model: ${GREEN}SmolVLM2-2.2B-Instruct${NC}"
echo -e "  Generator: ${GREEN}baseline${NC}"
if [ $# -gt 0 ]; then
    echo -e "  CLI overrides: ${YELLOW}$@${NC}"
fi
echo ""

# Run the evaluator
echo -e "${BLUE}🚀 Starting PROSPECT evaluation...${NC}"
echo -e "${BLUE}📂 Running from: $(pwd)${NC}"
echo -e "${BLUE}🐍 Python module: prospect.prospect_evaluator${NC}"
echo ""

run_python

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ PROSPECT evaluation completed successfully!${NC}"
    echo -e "${GREEN}📁 Check outputs in: custom/outputs/prospect/${NC}"
else
    echo ""
    echo -e "${RED}❌ PROSPECT evaluation failed!${NC}"
    exit 1
fi
