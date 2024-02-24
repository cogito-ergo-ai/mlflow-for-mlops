#!/bin/bash

curl -X 'POST' \
  'http://localhost:5000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    [
      0,
      0,
      0,
      0
    ],
[
      2.3,
      1.0,
      3.4,
      6.7
    ]
  ]
}'
