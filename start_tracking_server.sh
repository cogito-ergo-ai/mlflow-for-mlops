#!/bin/bash

pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 8080 -w 3
