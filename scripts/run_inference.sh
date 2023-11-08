#!/bin/bash
while read model_at; do
    python run_inference.py --model_at="$model_at"
done < models_list.txt