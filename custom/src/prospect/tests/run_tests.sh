#!/bin/bash

# Shell script to run PROSPECT tests
# Usage: 
#   ./run_tests.sh [all|custom_model|integration|strategy|quick]  - Run test suites
#   ./run_tests.sh e2e [none|drop_all|drop_middle|summarize_and_drop]  - Run single strategy E2E test
# Default: all

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PROSPECT Test Runner                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
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

# Activate conda environment
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}🔧 Found conda environment at ${VENV_PATH}${NC}"
    
    # Activate conda environment
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)" || true
        
        if conda activate "$VENV_PATH" 2>/dev/null; then
            echo -e "${GREEN}🔁 Activated conda environment${NC}"
        else
            echo -e "${YELLOW}⚠️  Using conda run -p instead${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  'conda' command not found, using venv python directly${NC}"
    fi
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

# Check if first argument is "e2e" for single strategy test
if [ "$1" = "e2e" ]; then
    # Single strategy E2E test mode
    STRATEGY="${2:-none}"
    
    # Validate strategy
    case "$STRATEGY" in
        none|drop_all|drop_middle|summarize_and_drop|summarize_with_dst)
            echo -e "${BLUE}Configuration:${NC}"
            echo -e "  Mode: ${YELLOW}Single Strategy E2E Test${NC}"
            echo -e "  Strategy: ${YELLOW}${STRATEGY}${NC}"
            echo ""
            ;;
        *)
            echo -e "${RED}❌ Invalid strategy: ${STRATEGY}${NC}"
            echo -e "${YELLOW}Valid strategies: none, drop_all, drop_middle, summarize_and_drop, summarize_with_dst${NC}"
            echo -e "${YELLOW}Usage: ./run_tests.sh e2e [strategy]${NC}"
            exit 1
            ;;
    esac
    
    # Run single strategy test
    echo -e "${BLUE}🚀 Starting E2E test for strategy: ${STRATEGY}...${NC}"
    echo ""
    
    # Python test will create timestamped output directory with logs and HTML
    "$PYTHON_CMD" "$PROJECT_ROOT/custom/src/prospect/tests/test_single_strategy.py" \
        --strategy "$STRATEGY" 2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Strategy ${STRATEGY} test completed successfully!${NC}"
        echo -e "${GREEN}� Output directory: custom/outputs/single_strategy_tests/${STRATEGY}/[timestamp]/${NC}"
    else
        echo ""
        echo -e "${RED}❌ Strategy ${STRATEGY} test failed!${NC}"
        exit $EXIT_CODE
    fi
    
else
    # Regular test suite mode
    TEST_SUITE="${1:-all}"
    
    # Show configuration
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  Mode: ${YELLOW}Test Suite${NC}"
    echo -e "  Suite: ${YELLOW}${TEST_SUITE}${NC}"
    echo -e "  Test directory: ${YELLOW}custom/src/prospect/tests/${NC}"
    echo ""
    
    # Check if pytest is installed
    if ! "$PYTHON_CMD" -c "import pytest" 2>/dev/null; then
        echo -e "${RED}❌ pytest is not installed!${NC}"
        echo -e "${YELLOW}Install it with: pip install pytest${NC}"
        exit 1
    fi
    
    # Run the test runner
    echo -e "${BLUE}🚀 Starting PROSPECT tests...${NC}"
    echo ""
    
    "$PYTHON_CMD" "$PROJECT_ROOT/custom/src/prospect/tests/run_tests.py" --suite "$TEST_SUITE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ All tests completed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}❌ Tests failed!${NC}"
        exit $EXIT_CODE
    fi
fi
