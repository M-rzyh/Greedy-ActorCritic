#!/bin/bash

# Configuration
ENV_CONFIG="config/environment/MountainCarContinuous-v1.json"
AGENT_CONFIG="config/agent/GreedyAC.json"
START_INDEX=3
END_INDEX=25
SAVE_DIR="./results"

# Loop through indices
for index in $(seq $START_INDEX $END_INDEX); do
    echo "========================================="
    echo "Running index $index of $END_INDEX"
    echo "========================================="
    
    python main.py \
        --env-json "$ENV_CONFIG" \
        --agent-json "$AGENT_CONFIG" \
        --index $index \
        --save-dir "$SAVE_DIR"
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Index $index completed successfully"
    else
        echo "ERROR: Index $index failed!"
        exit 1
    fi
    
    echo ""
done

echo "========================================="
echo "All runs completed!"
echo "========================================="
