#!/bin/bash

BASE_AGENT_JSON="config/agent/SAC_expectile.json"
ENV_JSON="config/environment/AcrobotContinuous-v1.json"
SAVE_DIR="results/expectile_sweep"

EXPECTILES=(0.7 0.8 0.9)
N=25
INDEXES=$(seq 1 $N)

# Create base save directory
mkdir -p "$SAVE_DIR"

for expectile in "${EXPECTILES[@]}"; do
    SAVE_SUBDIR="${SAVE_DIR}/expectile_${expectile}_n${N}"
    mkdir -p "$SAVE_SUBDIR"

    for index in $INDEXES; do
        NEW_JSON="config/agent/tmp_SAC_expectile_${expectile}_n${N}.json"

        jq --arg exp "$expectile" '.parameters.expectile = [$exp | tonumber]' \
           "$BASE_AGENT_JSON" > "$NEW_JSON"

        echo "Running expectile=$expectile, index=$index ..."
        python main.py \
            --agent-json "$NEW_JSON" \
            --env-json "$ENV_JSON" \
            --index "$index" \
            --save-dir "$SAVE_SUBDIR"

        if [ $? -ne 0 ]; then
            echo "Run failed for expectile=$expectile index=$index"
            exit 1
        fi
    done
done

echo "All experiments completed successfully!"
