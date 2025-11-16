#!/bin/bash

# Shell script to run DST Training Test
# Usage: ./run_dst_training_test.sh [--skip_model_loading]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DST Training Test Runner            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Project root is four levels up from custom/src/prospect/tests
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Project root directory: ${PROJECT_ROOT}${NC}"
echo -e "${GREEN}📁 Current working directory: $(pwd)${NC}"

# Activate conda environment
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}🔧 Found conda environment at ${VENV_PATH}${NC}"
else
    echo -e "${RED}❌ Error: Conda environment not found at ${VENV_PATH}${NC}"
    echo -e "${YELLOW}Please create it or check the path${NC}"
    exit 1
fi

# Add custom/src to PYTHONPATH for module imports
export PYTHONPATH="$PROJECT_ROOT/custom/src:$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}📦 PYTHONPATH: ${PYTHONPATH}${NC}"

# Build the Python command
PYTHON_CMD="$VENV_PATH/bin/python"
TEST_SCRIPT="$PROJECT_ROOT/custom/src/prospect/tests/test_dst_training.py"

# Parse arguments
SKIP_MODEL=""
if [[ "$1" == "--skip_model_loading" ]]; then
    SKIP_MODEL="--skip_model_loading"
    echo -e "${YELLOW}⏭️  Skipping model loading (faster test)${NC}"
fi

echo ""
echo -e "${BLUE}🚀 Running DST training setup test...${NC}"

# Run the test
"$PYTHON_CMD" "$TEST_SCRIPT" $SKIP_MODEL

# Check exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ DST training setup test completed successfully!${NC}"
    echo -e "${GREEN}🎯 Ready to run full training with: ./custom/runner/run_dst_training.sh${NC}"
else
    echo ""
    echo -e "${RED}❌ DST training setup test failed!${NC}"
    exit $EXIT_CODE
fi