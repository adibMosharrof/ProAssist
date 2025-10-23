#!/bin/bash

# Shell script to run the Simple DST Generator
# Usage: ./run_dst_generator.sh
# Configuration and overrides should be provided via Hydra configs only

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Simple DST Generator Runner         ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo ""

# Try to load common user shell profile files so environment variables (like OPENAI_API_KEY) are available
# Source multiple possible files (.bash_profile, .bashrc, .profile, .bash_login)
profile_files=("$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_login")
for pf in "${profile_files[@]}"; do
    if [ -f "$pf" ]; then
        # shellcheck source=/dev/null
        source "$pf" || true
        echo "Sourced: $pf"
    fi
done

# If a profile script changed HOME (some clusters do this), attempt to source the
# profile in the new HOME as well (e.g. initial HOME=/homes/adib sets HOME to /u/siddique-d1/adib)
if [ "$HOME" != "$LOGNAME" ] || true; then
    # Resolve the user's home directory explicitly and source their dotfiles again
    NEW_HOME="$HOME"
    # Only try if it's different from the original expanded path of ~ (some shells keep it)
    if [ -d "$NEW_HOME" ]; then
        pf2="$NEW_HOME/.bash_profile"
        if [ -f "$pf2" ]; then
            # shellcheck source=/dev/null
            source "$pf2" || true
            echo "Sourced (after HOME change): $pf2"
        fi
    fi
fi

# Show a masked presence check for OPENAI_API_KEY to help debug missing key issues
if [ -n "$OPENAI_API_KEY" ]; then
    # Mask the key (first 6 and last 4 chars)
    key_masked="${OPENAI_API_KEY:0:6}...${OPENAI_API_KEY: -4}"
    echo -e "${GREEN}🔑 OPENAI_API_KEY found: ${key_masked}${NC}"
else
    echo -e "${YELLOW}⚠️ OPENAI_API_KEY not found in environment after sourcing common profile files.${NC}"
    echo -e "${YELLOW}Check where you export the key (e.g. ~/.bashrc, ~/.bash_profile) and ensure it's exported with 'export OPENAI_API_KEY=...'.${NC}"
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ Error: OPENAI_API_KEY environment variable is not set${NC}"
    echo -e "${YELLOW}Please set it using: export OPENAI_API_KEY='your-api-key' or add it to ~/.bash_profile${NC}"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Project root is the repository root (one level up from custom/runner)
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
            echo -e "${YELLOW}⚠️ Could not 'conda activate' path; will use 'conda run -p' when executing python${NC}"
            ACTIVATED_VIA_ACTIVATE=0
        fi
    else
        echo -e "${YELLOW}⚠️ 'conda' command not found in PATH. Will attempt to run python from the venv directly if available.${NC}"
        ACTIVATED_VIA_ACTIVATE=0
    fi
else
    echo -e "${RED}❌ Error: Conda environment not found at ${VENV_PATH}${NC}"
    echo -e "${YELLOW}Please create it (e.g. conda create -p ${VENV_PATH} python=3.10) or update the path in the script${NC}"
    exit 1
fi

# Add custom/src to PYTHONPATH for module imports
export PYTHONPATH="$PROJECT_ROOT/custom/src:$PYTHONPATH"
echo -e "${GREEN}📦 PYTHONPATH: ${PYTHONPATH}${NC}"

# Build the Python command - run as module
PYTHON_CMD_ARGS="-m dst_data_builder.simple_dst_generator"

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
            echo -e "${RED}❌ No python executable found to run the generator${NC}"
            exit 1
        fi
    fi
}

# Show configuration (we don't accept CLI args; configs must come from Hydra)
echo ""
echo -e "${BLUE}Configuration: (from Hydra configs)${NC}"
echo -e "  Input directory: ${YELLOW}(from config file)${NC}"
echo -e "  Generator type: ${YELLOW}(from config file)${NC}"
echo -e "  Model: ${GREEN}gpt-4o${NC}"
echo ""

# Run the generator (no CLI overrides; Hydra config controls behavior)
echo -e "${BLUE}🚀 Starting DST generation (Hydra-controlled)...${NC}"
echo -e "${BLUE}📂 Running from: $(pwd)${NC}"
echo -e "${BLUE}🐍 Python module: dst_data_builder.simple_dst_generator ${NC}"
echo ""

run_python

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ DST generation completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ DST generation failed!${NC}"
    exit 1
fi
