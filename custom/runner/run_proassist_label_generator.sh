#!/bin/bash

#######################################################
# ProAssist DST Label Generator Runner
# Uses semantic alignment (bi-encoder + NLI) to generate
# step timestamps from procedural annotations
#######################################################

set -e

# Print header
echo "╔══════════════════════════════════════════════╗"
echo "║   ProAssist DST Label Generator Runner       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Source environment
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

# Check API keys
if [ -n "$OPENAI_API_KEY" ]; then
    KEY_DISPLAY="${OPENAI_API_KEY:0:6}...${OPENAI_API_KEY: -4}"
    echo "🔑 OPENAI_API_KEY found: $KEY_DISPLAY"
fi

# Get project root - run from /u/siddique-d1/adib/ProAssist
PROJECT_ROOT="/u/siddique-d1/adib/ProAssist"
if [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(pwd)"
fi
echo "📁 Project root directory: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"
echo "📁 Current working directory: $(pwd)"

# Look for venv
PYTHON_CMD="python3"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "🔧 Found virtual environment at $PROJECT_ROOT/.venv"
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
    if [ -f "$VENV_PYTHON" ]; then
        PYTHON_CMD="$VENV_PYTHON"
        echo "✓ Using Python from venv: $VENV_PYTHON"
    else
        echo "⚠️  Python not found in venv at $VENV_PYTHON, falling back to system python3"
    fi
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/custom/src:/mounts/u-amo-d1/adibm-data/projects/ZSToD/src:${PYTHONPATH:-}"
echo "📦 PYTHONPATH: $PYTHONPATH"
echo ""

echo "Configuration: (from Hydra configs)"
echo "  Input directory: (from config file)"
echo "  Generator type: proassist_label"
echo "  Model: gpt-4o (for reference only; main pipeline is deterministic)"
echo ""

echo "🚀 Starting ProAssist Label Generation (Hydra-controlled)..."
echo "📂 Running from: $(pwd)"
echo "🐍 Python command: $PYTHON_CMD"
echo "🐍 Python module: dst_data_builder.simple_dst_generator"
echo ""

# Run the generator
cd "$PROJECT_ROOT"

$PYTHON_CMD -m dst_data_builder.simple_dst_generator 
echo ""
echo "✅ ProAssist DST label generation completed successfully!"
