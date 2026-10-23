#!/bin/bash
cd "$(dirname "$0")/backend" && pip install -r requirements.txt && uvicorn main:app --host 127.0.0.1 --port 8000
