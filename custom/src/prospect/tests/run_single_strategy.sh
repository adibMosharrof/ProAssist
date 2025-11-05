#!/bin/bash

# Shell script to run a single strategy E2E test
# Usage: ./run_single_strategy.sh [none|drop_all|drop_middle|summarize_and_drop]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PROSPECT Single Strategy Test       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if strategy argument provided
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Strategy argument required${NC}"
    echo -e "${YELLOW}Usage: ./run_single_strategy.sh [none|drop_all|drop_middle|summarize_and_drop]${NC}"
    exit 1
fi

STRATEGY="$1"

# Validate strategy
case "$STRATEGY" in
    none|drop_all|drop_middle|summarize_and_drop|summarize_with_dst)
        echo -e "${GREEN}✅ Testing strategy: ${STRATEGY}${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid strategy: ${STRATEGY}${NC}"
        echo -e "${YELLOW}Valid strategies: none, drop_all, drop_middle, summarize_and_drop, summarize_with_dst${NC}"
        exit 1
        ;;
esac

echo ""

# Source bash profile to set correct HOME directory and environment
# This fixes disk space issues related to incorrect HOME directory
if [ -f ~/.bash_profile ]; then
    source ~/.bash_profile
    echo -e "${GREEN}✅ Sourced ~/.bash_profile (HOME: $HOME)${NC}"
else
    echo -e "${YELLOW}⚠️  ~/.bash_profile not found, using default HOME${NC}"
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Project root is three levels up from custom/src/prospect/tests
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Project root directory: ${PROJECT_ROOT}${NC}"
echo -e "${GREEN}📁 Current working directory: $(pwd)${NC}"
echo -e "${GREEN}🏠 HOME directory: $HOME${NC}"
echo ""

# Use venv python
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}🔧 Found conda environment at ${VENV_PATH}${NC}"
else
    echo -e "${RED}❌ Error: Conda environment not found at ${VENV_PATH}${NC}"
    exit 1
fi

# Add custom/src to PYTHONPATH for module imports
export PYTHONPATH="$PROJECT_ROOT/custom/src:$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}📦 PYTHONPATH: ${PYTHONPATH}${NC}"
echo ""

# Use venv python
PYTHON_CMD="$VENV_PATH/bin/python"

# Run the single strategy test
echo -e "${BLUE}🚀 Starting test for strategy: ${STRATEGY}...${NC}"
echo ""

# Run test (Python handles output directory and logging)
"$PYTHON_CMD" "$PROJECT_ROOT/custom/src/prospect/tests/test_single_strategy.py" \
    --strategy "$STRATEGY"

# Check exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Strategy ${STRATEGY} test completed successfully!${NC}"
    echo -e "${GREEN}� Output saved to: custom/outputs/single_strategy_tests/${STRATEGY}/<timestamp>/${NC}"
else
    echo ""
    echo -e "${RED}❌ Strategy ${STRATEGY} test failed!${NC}"
    exit $EXIT_CODE
fi
