#!/bin/bash

pip install -r requirements.txt
uvicorn app:app --port 5000
