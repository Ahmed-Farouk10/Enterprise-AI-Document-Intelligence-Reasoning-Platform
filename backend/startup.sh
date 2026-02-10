#!/bin/bash
# Startup script for Hugging Face Spaces deployment
# This ensures all environment variables are set before the application starts

echo "===== Application Startup at $(date '+%Y-%m-%d %H:%M:%S') ====="
echo ""

# CRITICAL: Set LLM_API_KEY for Cognee compatibility
if [ -z "$LLM_API_KEY" ]; then
    if [ -n "$HF_TOKEN" ]; then
        export LLM_API_KEY="$HF_TOKEN"
        echo "🔑 LLM_API_KEY set from HF_TOKEN"
    else
        export LLM_API_KEY="local"
        echo "🔑 LLM_API_KEY set to 'local' (no HF_TOKEN found)"
    fi
else
    echo "🔑 LLM_API_KEY already set"
fi

# Display Cognee configuration
echo ""
echo "================================================================================"
echo "🧠 COGNEE CONFIGURATION (AGGRESSIVE)"
echo "================================================================================"
echo "Environment: HuggingFace Spaces (HF_HOME)"
echo "Cognee Root: ${HF_HOME}/cognee_data"
echo "SYSTEM_ROOT_DIRECTORY: ${HF_HOME}/cognee_data"
echo "DB Path: ${HF_HOME}/cognee_data/databases"
echo "Writable: $([ -w "${HF_HOME}/cognee_data" ] && echo 'True' || echo 'False')"
echo "Exists: $([ -d "${HF_HOME}/cognee_data" ] && echo 'True' || echo 'False')"
echo "LLM_API_KEY: ${LLM_API_KEY:0:10}... (truncated)"
echo "================================================================================"
echo ""

# Start the application
exec "$@"
