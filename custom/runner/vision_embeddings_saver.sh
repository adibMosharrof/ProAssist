#!/bin/bash

#######################################################
# Vision Embeddings Saver Runner
# Extracts and saves vision embeddings from SmolVLM2
# for DST training data preprocessing
#######################################################

set -e

# Print header
echo "╔══════════════════════════════════════════════╗"
echo "║   Vision Embeddings Saver Runner ║"
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


echo "🚀 Starting Vision Embeddings Extraction (Hydra-controlled)..."
echo "📂 Running from: $(pwd)"
echo "🐍 Python command: $PYTHON_CMD"
echo "🐍 Python module: dst_data_builder.vision_embeddings_runner"
echo ""

# Run the embeddings saver
cd "$PROJECT_ROOT"

$PYTHON_CMD -m dst_data_builder.vision_embeddings_saver
echo ""
echo "✅ Vision embeddings extraction completed successfully!"