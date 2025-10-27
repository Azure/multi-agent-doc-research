#!/bin/bash

# Chainlit file upload environment variables
export CHAINLIT_ALLOW_UPLOAD=true
export CHAINLIT_MAX_SIZE_MB=50
export CHAINLIT_MAX_FILES=10

# Run the Chainlit app
cd /home/azureuser/cloudfiles/code/doc-inquiry-chatbot/app/frontend
chainlit run src/app.py --host 0.0.0.0 --port 8501