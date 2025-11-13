#!/bin/bash

BASE_AGENT_JSON="config/agent/SAC.json"
ENV_JSON="config/environment/AcrobotContinuous-v1.json"
SAVE_DIR="no_expectile_sweep"

INDEXES=$(seq 1 25)

# Create save directory
mkdir -p "$SAVE_DIR"

for index in $INDEXES; do

    echo "Running index=$index ..."

    python main.py \
      --agent-json "$BASE_AGENT_JSON" \
      --env-json "$ENV_JSON" \
      --index "$index" \
      --save-dir "$SAVE_DIR"

    if [ $? -ne 0 ]; then
        echo "Run failed for index=$index"
        exit 1
    fi

done

echo "All 25 runs completed successfully!"
