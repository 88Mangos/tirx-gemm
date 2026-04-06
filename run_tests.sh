#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --step <step1,step2,...>"
    echo "Example: $0 --step 1,2,4"
    echo "Valid steps: 1,2,3,4,5,6,7,8,9,10"
    exit 1
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Parse command line arguments
STEPS=""
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate steps
if [ -z "$STEPS" ]; then
    echo "Error: --step argument is required"
    usage
fi

# Validate that steps contain only numbers and commas
if ! [[ "$STEPS" =~ ^[0-9,]+$ ]]; then
    echo "Error: Steps must contain only numbers and commas (e.g., 1,2,4)"
    exit 1
fi

# Convert steps to array and validate range
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"
for step in "${STEP_ARRAY[@]}"; do
    if ! [[ "$step" =~ ^[0-9]+$ ]] || [ "$step" -lt 1 ] || [ "$step" -gt 10 ]; then
        echo "Error: Invalid step '$step'. Valid steps are 1-10"
        exit 1
    fi
done

# Function to activate virtual environment
activate_venv() {
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        echo "Activated virtual environment: $VENV_PATH"
    elif [ -f "$VENV_PATH/Scripts/activate" ]; then
        source "$VENV_PATH/Scripts/activate"
        echo "Activated virtual environment: $VENV_PATH"
    else
        echo "Error: Virtual environment not found at $VENV_PATH"
        echo "Expected: $VENV_PATH/bin/activate or $VENV_PATH/Scripts/activate"
        exit 1
    fi
}

# Main execution
echo "Running Modal tests for steps: $STEPS"
echo ""

# Activate virtual environment first
echo "Activating virtual environment..."
activate_venv

# Create output file with timestamp if not provided
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="modal_output_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Run all steps together and capture output
echo "Running all steps simultaneously..."
{
    echo "========================================"
    echo "Modal run output - Started at $(date)"
    echo "Steps: $STEPS"
    echo "========================================"
    echo "Running: modal run run_modal.py --step $STEPS"
    echo ""
    
    # Run the command directly - this should preserve colors
    (source "$VENV_PATH/bin/activate" && modal run run_modal.py --step "$STEPS")
    
    echo ""
    echo "========================================"
    echo "Run completed at $(date)"
} | tee -a "$OUTPUT_FILE"

echo ""
echo "Output saved to: $OUTPUT_FILE"
echo ""
echo "=== Terminal Output ==="
cat "$OUTPUT_FILE"
