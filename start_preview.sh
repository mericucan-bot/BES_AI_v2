#!/bin/bash
export PYTHONPATH="/Users/mericucan/Desktop/BES_AI_v2/.venv/lib/python3.9/site-packages:$PYTHONPATH"
export VIRTUAL_ENV=""
export DEV_BYPASS_AUTH=true
cd "/Users/mericucan/AI Projects/BES_AI_v2"
/usr/bin/python3 -m streamlit run app.py --server.headless true --server.port ${PORT:-8502}
